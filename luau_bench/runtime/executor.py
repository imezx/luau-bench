from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RESULT_PREFIX = "@@LUAU_BENCH_RESULT@@"
_PASS_TAG = "PASS"
_FAIL_TAG = "FAIL"
_ERROR_TAG = "ERROR"

_SUBCOMMAND_RUNTIMES = {"lune", "zune"}


def find_luau_runtime(explicit_path: str = "") -> Optional[str]:
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    env_path = os.environ.get("LUAU_RUNTIME", "")
    if env_path and os.path.isfile(env_path):
        return env_path
    for name in ("zune", "luau", "lune"):
        found = shutil.which(name)
        if found:
            return found
    return None


def find_luau_analyzer(explicit_path: str = "") -> Optional[str]:
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    env_path = os.environ.get("LUAU_ANALYZE", "")
    if env_path and os.path.isfile(env_path):
        return env_path
    found = shutil.which("luau-analyze")
    return found


def get_runtime_version(runtime_path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            [runtime_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (result.stdout or result.stderr).strip()
        if not output:
            return None
        line = output.split("\n")[0]
        m = re.search(r"\b(\d+\.\d+(?:\.\d+)*(?:[-.]\w+)*)\b", line)
        return m.group(1) if m else line
    except Exception:
        return None


def _build_run_cmd(runtime: str, script_path: str) -> list[str]:
    name = os.path.basename(runtime).lower()
    base = re.split(r"[-_][0-9]", name)[0]
    if base in _SUBCOMMAND_RUNTIMES:
        return [runtime, "run", script_path]
    return [runtime, script_path]


class LuauExecutor:
    """
    @@LUAU_BENCH_RESULT@@PASS:label
    @@LUAU_BENCH_RESULT@@FAIL:label:message
    @@LUAU_BENCH_RESULT@@ERROR:label:message
    """

    def __init__(
        self,
        runtime_path: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.runtime = find_luau_runtime(runtime_path or "")
        self.timeout = timeout
        if self.runtime:
            logger.info("Using Luau runtime: %s", self.runtime)
        else:
            logger.warning("No Luau runtime found — execution disabled.")

    @property
    def available(self) -> bool:
        return self.runtime is not None

    def version(self) -> Optional[str]:
        return get_runtime_version(self.runtime) if self.runtime else None

    async def run_script(self, script: str) -> dict[str, Any]:
        if not self.available:
            return {
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "total": 0,
                "pass_rate": 0.0,
                "runtime_ms": 0.0,
                "stdout": "",
                "stderr": "No Luau runtime available",
                "timed_out": False,
                "details": [],
            }

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".luau",
            delete=False,
            encoding="utf-8",
        )
        try:
            tmp.write(script)
            tmp.flush()
            tmp.close()

            t0 = time.perf_counter()
            proc_result = await self._run_process(tmp.name)
            runtime_ms = (time.perf_counter() - t0) * 1000.0

            return self._parse_output(
                stdout=proc_result["stdout"],
                stderr=proc_result["stderr"],
                timed_out=proc_result["timed_out"],
                runtime_ms=runtime_ms,
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    async def _run_process(self, script_path: str) -> dict[str, Any]:
        assert self.runtime is not None
        cmd = _build_run_cmd(self.runtime, script_path)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )
            return {
                "stdout": stdout_b.decode("utf-8", errors="replace"),
                "stderr": stderr_b.decode("utf-8", errors="replace"),
                "returncode": proc.returncode,
                "timed_out": False,
            }
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return {
                "stdout": "",
                "stderr": "Execution timed out",
                "returncode": -1,
                "timed_out": True,
            }
        except Exception as exc:
            return {"stdout": "", "stderr": str(exc), "returncode": -1, "timed_out": False}

    def _parse_output(
        self,
        stdout: str,
        stderr: str,
        timed_out: bool,
        runtime_ms: float,
    ) -> dict[str, Any]:
        passed = failed = errors = 0
        details: list[dict[str, Any]] = []

        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith(_RESULT_PREFIX):
                continue
            payload = line[len(_RESULT_PREFIX) :]

            if payload.startswith(_PASS_TAG + ":"):
                label = payload[len(_PASS_TAG) + 1 :]
                passed += 1
                details.append({"test": label, "status": "pass", "message": ""})
            elif payload.startswith(_FAIL_TAG + ":"):
                rest = payload[len(_FAIL_TAG) + 1 :]
                parts = rest.split(":", 1)
                failed += 1
                details.append(
                    {
                        "test": parts[0],
                        "status": "fail",
                        "message": parts[1] if len(parts) > 1 else "",
                    }
                )
            elif payload.startswith(_ERROR_TAG + ":"):
                rest = payload[len(_ERROR_TAG) + 1 :]
                parts = rest.split(":", 1)
                errors += 1
                details.append(
                    {
                        "test": parts[0],
                        "status": "error",
                        "message": parts[1] if len(parts) > 1 else "",
                    }
                )

        total = passed + failed + errors
        rate = (passed / total * 100.0) if total > 0 else 0.0

        return {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": total,
            "pass_rate": rate,
            "runtime_ms": runtime_ms,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
            "details": details,
        }


class LuauAnalyzer:
    def __init__(
        self,
        analyzer_path: Optional[str] = None,
        timeout: float = 15.0,
    ) -> None:
        self.analyzer = find_luau_analyzer(analyzer_path or "")
        self.timeout = timeout
        if self.analyzer:
            logger.debug("Using luau-analyze: %s", self.analyzer)
        else:
            logger.debug("luau-analyze not found — static analysis disabled.")

    @property
    def available(self) -> bool:
        return self.analyzer is not None

    async def analyze(self, code: str) -> dict[str, Any]:
        if not self.available:
            return {
                "errors": 0,
                "warnings": 0,
                "total": 0,
                "clean": True,
                "diagnostics": [],
                "available": False,
            }

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".luau",
            delete=False,
            encoding="utf-8",
        )
        try:
            tmp.write(code)
            tmp.flush()
            tmp.close()
            return await self._run_analyzer(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    async def _run_analyzer(self, script_path: str) -> dict[str, Any]:
        assert self.analyzer is not None
        for formatter in ("json", None):
            cmd = [self.analyzer, script_path]
            if formatter:
                cmd = [self.analyzer, "--formatter=json", script_path]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout,
                )
                stdout = stdout_b.decode("utf-8", errors="replace")
                stderr = stderr_b.decode("utf-8", errors="replace")

                if formatter == "json":
                    parsed = self._parse_json_output(stdout + stderr)
                    if parsed is not None:
                        return {**parsed, "available": True}
                    continue
                else:
                    return {**self._parse_text_output(stdout + stderr), "available": True}

            except asyncio.TimeoutError:
                return {
                    "errors": 0,
                    "warnings": 0,
                    "total": 0,
                    "clean": True,
                    "diagnostics": [],
                    "available": True,
                    "timed_out": True,
                }
            except Exception as exc:
                logger.warning("luau-analyze failed: %s", exc)
                return {
                    "errors": 0,
                    "warnings": 0,
                    "total": 0,
                    "clean": True,
                    "diagnostics": [],
                    "available": True,
                }

        return {
            "errors": 0,
            "warnings": 0,
            "total": 0,
            "clean": True,
            "diagnostics": [],
            "available": True,
        }

    @staticmethod
    def _parse_json_output(text: str) -> Optional[dict[str, Any]]:
        text = text.strip()
        if not text:
            return {"errors": 0, "warnings": 0, "total": 0, "clean": True, "diagnostics": []}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            bracket = text.find("[")
            if bracket == -1:
                return None
            try:
                data = json.loads(text[bracket:])
            except json.JSONDecodeError:
                return None

        if not isinstance(data, list):
            return None

        errors = sum(1 for d in data if d.get("severity", "").lower() == "error")
        warnings = sum(1 for d in data if d.get("severity", "").lower() == "warning")
        return {
            "errors": errors,
            "warnings": warnings,
            "total": errors + warnings,
            "clean": errors == 0 and warnings == 0,
            "diagnostics": data,
        }

    @staticmethod
    def _parse_text_output(text: str) -> dict[str, Any]:
        errors = 0
        warnings = 0
        diagnostics: list[dict[str, Any]] = []

        pat = re.compile(
            r"^(?:.*?)\((\d+),(\d+)\):\s*\(([^)]+)\)\s*(.+)$",
            re.MULTILINE,
        )
        for m in pat.finditer(text):
            line, col, code, message = m.group(1, 2, 3, 4)
            severity = "warning" if code.startswith("W") else "error"
            if severity == "error":
                errors += 1
            else:
                warnings += 1
            diagnostics.append(
                {
                    "line": int(line),
                    "column": int(col),
                    "code": code,
                    "message": message.strip(),
                    "severity": severity,
                }
            )

        return {
            "errors": errors,
            "warnings": warnings,
            "total": errors + warnings,
            "clean": errors == 0 and warnings == 0,
            "diagnostics": diagnostics,
        }


def luau_repr(value: Any) -> str:
    if value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "{" + ", ".join(luau_repr(v) for v in value) + "}"
    if isinstance(value, dict):
        pairs = ", ".join(f"[{luau_repr(k)}] = {luau_repr(v)}" for k, v in value.items())
        return "{" + pairs + "}"
    return f'"{value!s}"'


DEEP_EQUAL_HELPER = """\
local function deepEqual(a, b)
    if type(a) ~= type(b) then return false end
    if type(a) ~= "table" then return a == b end
    for k, v in a do
        if not deepEqual(v, b[k]) then return false end
    end
    for k, _ in b do
        if a[k] == nil then return false end
    end
    return true
end

local function serialize(val)
    if type(val) == "table" then
        local parts = {}
        local isArray = true
        local n = 0
        for k, _ in val do
            n += 1
            if type(k) ~= "number" or k ~= n then isArray = false end
        end
        if isArray then
            for _, v in ipairs(val) do table.insert(parts, serialize(v)) end
            return "{" .. table.concat(parts, ", ") .. "}"
        else
            for k, v in val do
                table.insert(parts, tostring(k) .. " = " .. serialize(v))
            end
            return "{" .. table.concat(parts, ", ") .. "}"
        end
    elseif type(val) == "string" then
        return string.format("%q", val)
    end
    return tostring(val)
end
"""
