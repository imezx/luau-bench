from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StyLuaResult:
    available: bool = False
    parseable: bool = False
    already_formatted: bool = False
    formatted_code: str = ""
    diff_lines: int = 0
    total_lines: int = 0
    error_message: str = ""

    @property
    def format_match_ratio(self) -> float:
        if self.total_lines == 0:
            return 1.0
        return max(0.0, 1.0 - (self.diff_lines / self.total_lines))


def find_stylua(explicit_path: str = "") -> Optional[str]:
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    env_path = os.environ.get("STYLUA_PATH", "")
    if env_path and os.path.isfile(env_path):
        return env_path
    return shutil.which("stylua")


def get_stylua_version(stylua_path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            [stylua_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (result.stdout or result.stderr).strip()
        return output.split("\n")[0] if output else None
    except Exception:
        return None


class StyLuaChecker:
    def __init__(self, stylua_path: str = "") -> None:
        self.binary = find_stylua(stylua_path)

    @property
    def available(self) -> bool:
        return self.binary is not None

    async def check(self, code: str) -> StyLuaResult:
        if not self.available:
            return StyLuaResult(available=False)

        total_lines = len(code.strip().split("\n")) if code.strip() else 0
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

            formatted, error_msg, returncode = await self._run_stylua(tmp.name)
            if returncode != 0 or error_msg:
                return StyLuaResult(
                    available=True,
                    parseable=False,
                    total_lines=total_lines,
                    error_message=error_msg,
                )

            orig_lines = code.strip().split("\n")
            fmt_lines = formatted.strip().split("\n")
            diff_count = sum(
                1
                for i in range(max(len(orig_lines), len(fmt_lines)))
                if (orig_lines[i].rstrip() if i < len(orig_lines) else "")
                != (fmt_lines[i].rstrip() if i < len(fmt_lines) else "")
            )

            return StyLuaResult(
                available=True,
                parseable=True,
                already_formatted=diff_count == 0,
                formatted_code=formatted,
                diff_lines=diff_count,
                total_lines=total_lines,
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    async def _run_stylua(self, file_path: str) -> tuple[str, str, int]:
        assert self.binary is not None
        try:
            proc = await asyncio.create_subprocess_exec(
                self.binary,
                file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            if proc.returncode != 0:
                return "", stderr.decode("utf-8", errors="replace").strip(), proc.returncode or 1
            with open(file_path, "r", encoding="utf-8") as f:
                formatted = f.read()
            return formatted, "", 0
        except asyncio.TimeoutError:
            return "", "StyLua timed out", -1
        except Exception as exc:
            return "", str(exc), -1
