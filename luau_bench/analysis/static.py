from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class LuauAnalysis:
    total_lines: int = 0
    code_lines: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    total_chars: int = 0

    cyclomatic_complexity: int = 1
    max_nesting_depth: int = 0
    avg_nesting_depth: float = 0.0

    typed_params: int = 0
    untyped_params: int = 0
    typed_returns: int = 0
    untyped_returns: int = 0
    type_aliases: int = 0
    has_strict_mode: bool = False
    has_nonstrict_mode: bool = False

    # Functions
    function_count: int = 0
    local_function_count: int = 0
    global_function_count: int = 0
    anonymous_function_count: int = 0
    avg_function_length: float = 0.0

    local_var_count: int = 0
    global_var_count: int = 0

    api_calls: list[str] = field(default_factory=list)

    if_count: int = 0
    elseif_count: int = 0
    for_count: int = 0
    while_count: int = 0
    repeat_count: int = 0
    return_count: int = 0
    break_count: int = 0
    continue_count: int = 0

    pcall_count: int = 0
    xpcall_count: int = 0
    error_call_count: int = 0
    assert_count: int = 0

    uses_if_expression: bool = False
    uses_string_interpolation: bool = False
    uses_compound_assignment: bool = False
    uses_generics: bool = False
    uses_union_types: bool = False
    uses_intersection_types: bool = False
    uses_optional_types: bool = False
    uses_typeof: bool = False
    uses_table_freeze: bool = False
    uses_table_clone: bool = False
    uses_const: bool = False  # 711+
    const_var_count: int = 0

    trailing_whitespace_lines: int = 0
    mixed_indentation: bool = False
    max_line_length: int = 0
    lines_over_120: int = 0

    code_smells: list[str] = field(default_factory=list)

    @property
    def type_coverage(self) -> float:
        total = self.typed_params + self.untyped_params
        return (self.typed_params / total * 100.0) if total > 0 else 100.0

    @property
    def comment_ratio(self) -> float:
        return (self.comment_lines / self.code_lines * 100.0) if self.code_lines > 0 else 0.0

    @property
    def locality_ratio(self) -> float:
        total = self.local_var_count + self.global_var_count
        return (self.local_var_count / total * 100.0) if total > 0 else 100.0


# regex

_STRICT_MODE = re.compile(r"^--!strict\s*$", re.MULTILINE)
_NONSTRICT_MODE = re.compile(r"^--!nonstrict\s*$", re.MULTILINE)
_TYPE_ALIAS = re.compile(r"^\s*(?:export\s+)?type\s+\w+", re.MULTILINE)
_RETURN_TYPE = re.compile(r"\)\s*:\s*[A-Za-z_{(]")
_PARAM_UNTYPED = re.compile(r"(?:function\s+\w+|function)\s*\(([^)]*)\)")

_LOCAL_FUNC = re.compile(r"^\s*local\s+function\s+(\w+)", re.MULTILINE)
_GLOBAL_FUNC = re.compile(r"^function\s+(\w+)", re.MULTILINE)
_ANON_FUNC = re.compile(r"=\s*function\s*\(")
_LOCAL_VAR = re.compile(r"^\s*local\s+(\w+)", re.MULTILINE)
_CONST_DECL = re.compile(r"^\s*const\s+(\w+)", re.MULTILINE)  # 711+
_GLOBAL_ASSIGN = re.compile(r"^(\w+)\s*=(?!=)", re.MULTILINE)

_IF_KW = re.compile(r"\bif\b")
_ELSEIF_KW = re.compile(r"\belseif\b")
_FOR_KW = re.compile(r"\bfor\b")
_WHILE_KW = re.compile(r"\bwhile\b")
_REPEAT_KW = re.compile(r"\brepeat\b")
_AND_KW = re.compile(r"\band\b")
_OR_KW = re.compile(r"\bor\b")

_PCALL = re.compile(r"\bpcall\s*\(")
_XPCALL = re.compile(r"\bxpcall\s*\(")
_ERROR_CALL = re.compile(r"\berror\s*\(")
_ASSERT_CALL = re.compile(r"\bassert\s*\(")

_IF_EXPR = re.compile(r"=\s*if\b.+\bthen\b.+\belse\b")
_STRING_INTERP = re.compile(r"`[^`]*\{")
_COMPOUND_ASSIGN = re.compile(r"\b\w+\s*[+\-*/%]=")
_GENERICS = re.compile(r"<\s*\w+\s*>")
_UNION_TYPE = re.compile(r":\s*[^=\n]*\|")
_INTERSECTION_TYPE = re.compile(r":\s*[^=\n]*&")
_OPTIONAL_TYPE = re.compile(r":\s*\w+\?")
_TYPEOF = re.compile(r"\btypeof\s*\(")
_TABLE_FREEZE = re.compile(r"\btable\.freeze\s*\(")
_TABLE_CLONE = re.compile(r"\btable\.clone\s*\(")

_API_CALL = re.compile(
    r"\b(table\.\w+|string\.\w+|math\.\w+|coroutine\.\w+|"
    r"buffer\.\w+|bit32\.\w+|os\.\w+|debug\.\w+|"
    r"task\.\w+|Instance\.new|game\.\w+|workspace\.\w+|"
    r"Vector[23]\.new|CFrame\.new|Color3\.\w+|"
    r"UDim2?\.[\w+]|Enum\.\w+|"
    r"tonumber|tostring|type|typeof|select|unpack|"
    r"pairs|ipairs|next|rawget|rawset|rawequal|rawlen|"
    r"setmetatable|getmetatable|"
    r"print|warn|error|assert|pcall|xpcall|require)\b"
)

_KEYWORDS = {
    "if",
    "else",
    "elseif",
    "then",
    "do",
    "end",
    "for",
    "while",
    "repeat",
    "until",
    "return",
    "break",
    "continue",
    "function",
    "local",
    "const",
    "true",
    "false",
    "nil",
    "and",
    "or",
    "not",
    "in",
    "type",
    "export",
}


def _strip_strings_and_comments(code: str) -> str:
    result = list(code)
    i = 0
    n = len(code)

    while i < n:
        if code[i : i + 4] == "--[[":
            end = code.find("]]", i + 4)
            end = n if end == -1 else end + 2
            for j in range(i, end):
                if result[j] != "\n":
                    result[j] = " "
            i = end
            continue

        if code[i : i + 2] == "--":
            end = code.find("\n", i)
            end = n if end == -1 else end
            for j in range(i, end):
                result[j] = " "
            i = end
            continue

        if code[i] == "[" and i + 1 < n and code[i + 1] in ("[", "="):
            eq_count = 0
            j = i + 1
            while j < n and code[j] == "=":
                eq_count += 1
                j += 1
            if j < n and code[j] == "[":
                close = "]" + "=" * eq_count + "]"
                end = code.find(close, j + 1)
                end = n if end == -1 else end + len(close)
                for k in range(i, end):
                    if result[k] != "\n":
                        result[k] = " "
                i = end
                continue

        for quote in ('"', "'", "`"):
            if code[i] == quote:
                j = i + 1
                while j < n and code[j] != quote:
                    if code[j] == "\\":
                        j += 1
                    j += 1
                if j < n:
                    j += 1
                for k in range(i, j):
                    if result[k] != "\n":
                        result[k] = " "
                i = j
                break
        else:
            i += 1
            continue
        continue

    return "".join(result)


def analyze(code: str) -> LuauAnalysis:
    a = LuauAnalysis()
    lines = code.split("\n")
    a.total_lines = len(lines)
    a.total_chars = len(code)
    clean = _strip_strings_and_comments(code)

    indent_tabs = indent_spaces = 0
    nesting_depths: list[int] = []
    current_depth = 0

    for line in lines:
        stripped = line.rstrip()
        raw = line.rstrip("\n\r")

        if raw != raw.rstrip():
            a.trailing_whitespace_lines += 1
        ll = len(raw)
        if ll > a.max_line_length:
            a.max_line_length = ll
        if ll > 120:
            a.lines_over_120 += 1

        if not stripped:
            a.blank_lines += 1
        elif stripped.startswith("--"):
            a.comment_lines += 1
        else:
            a.code_lines += 1

        if stripped:
            if line.startswith("\t"):
                indent_tabs += 1
            elif line.startswith("  "):
                indent_spaces += 1

        for word in ("function", "if", "for", "while", "repeat", "do"):
            if re.search(rf"\b{word}\b", stripped) and not stripped.startswith("--"):
                current_depth += 1
        for word in ("end", "until"):
            if re.search(rf"\b{word}\b", stripped) and not stripped.startswith("--"):
                current_depth = max(0, current_depth - 1)
        nesting_depths.append(current_depth)

    a.max_nesting_depth = max(nesting_depths) if nesting_depths else 0
    a.avg_nesting_depth = sum(nesting_depths) / len(nesting_depths) if nesting_depths else 0.0
    a.mixed_indentation = indent_tabs > 0 and indent_spaces > 0

    a.has_strict_mode = bool(_STRICT_MODE.search(code))
    a.has_nonstrict_mode = bool(_NONSTRICT_MODE.search(code))
    a.type_aliases = len(_TYPE_ALIAS.findall(code))

    for m in _PARAM_UNTYPED.finditer(clean):
        for param in m.group(1).strip().split(","):
            param = param.strip()
            if not param or param == "...":
                continue
            if ":" in param:
                a.typed_params += 1
            else:
                a.untyped_params += 1

    a.typed_returns = len(_RETURN_TYPE.findall(clean))

    a.local_function_count = len(_LOCAL_FUNC.findall(clean))
    a.global_function_count = len(_GLOBAL_FUNC.findall(clean))
    a.anonymous_function_count = len(_ANON_FUNC.findall(clean))
    a.function_count = a.local_function_count + a.global_function_count + a.anonymous_function_count

    a.local_var_count = len(_LOCAL_VAR.findall(clean))
    a.const_var_count = len(_CONST_DECL.findall(clean))
    a.uses_const = a.const_var_count > 0
    a.local_var_count += a.const_var_count
    for m in _GLOBAL_ASSIGN.finditer(clean):
        if m.group(1) not in _KEYWORDS:
            a.global_var_count += 1

    for cl in clean.split("\n"):
        a.if_count += len(_IF_KW.findall(cl))
        a.elseif_count += len(_ELSEIF_KW.findall(cl))
        a.for_count += len(_FOR_KW.findall(cl))
        a.while_count += len(_WHILE_KW.findall(cl))
        a.repeat_count += len(_REPEAT_KW.findall(cl))

    a.return_count = len(re.findall(r"\breturn\b", clean))
    a.break_count = len(re.findall(r"\bbreak\b", clean))
    a.continue_count = len(re.findall(r"\bcontinue\b", clean))

    decision_points = (
        a.if_count
        + a.elseif_count
        + a.for_count
        + a.while_count
        + a.repeat_count
        + len(_AND_KW.findall(clean))
        + len(_OR_KW.findall(clean))
    )
    a.cyclomatic_complexity = 1 + decision_points

    a.pcall_count = len(_PCALL.findall(clean))
    a.xpcall_count = len(_XPCALL.findall(clean))
    a.error_call_count = len(_ERROR_CALL.findall(clean))
    a.assert_count = len(_ASSERT_CALL.findall(clean))

    a.uses_if_expression = bool(_IF_EXPR.search(clean))
    a.uses_string_interpolation = bool(_STRING_INTERP.search(code))
    a.uses_compound_assignment = bool(_COMPOUND_ASSIGN.search(clean))
    a.uses_generics = bool(_GENERICS.search(clean))
    a.uses_union_types = bool(_UNION_TYPE.search(clean))
    a.uses_intersection_types = bool(_INTERSECTION_TYPE.search(clean))
    a.uses_optional_types = bool(_OPTIONAL_TYPE.search(clean))
    a.uses_typeof = bool(_TYPEOF.search(clean))
    a.uses_table_freeze = bool(_TABLE_FREEZE.search(clean))
    a.uses_table_clone = bool(_TABLE_CLONE.search(clean))
    a.uses_const = a.const_var_count > 0

    a.api_calls = list(set(_API_CALL.findall(code)))

    if a.max_nesting_depth > 5:
        a.code_smells.append(f"Deep nesting (depth {a.max_nesting_depth})")
    if a.global_var_count > 0 and a.function_count > 0:
        a.code_smells.append(f"Global variable pollution ({a.global_var_count} globals)")
    if a.cyclomatic_complexity > 15:
        a.code_smells.append(f"High cyclomatic complexity ({a.cyclomatic_complexity})")
    if a.function_count > 0 and a.code_lines > 0:
        a.avg_function_length = a.code_lines / a.function_count

    return a


def check_code_validity(code: str) -> dict[str, object]:
    """
    Returns ``{"valid": bool, "confidence": float, "issues": list[str]}``.

    Confidence scale
    ----------------
    1.0  - strong positive signals, no issues found
    0.7+ - plausibly valid luau with minor concerns
    0.4-0.7 - uncertain, possible issues
    <0.4 - very likely not valid luau
    """
    issues: list[str] = []
    confidence = 1.0
    stripped = code.strip()

    if not stripped:
        return {"valid": False, "confidence": 0.0, "issues": ["Empty code"]}

    if len(stripped) < 10:
        issues.append(f"Extremely short code ({len(stripped)} chars)")
        confidence -= 0.5
        return {"valid": False, "confidence": max(0.0, confidence), "issues": issues}

    clean = _strip_strings_and_comments(code)

    openers = len(re.findall(r"\b(?:function|if|for|while|repeat|do)\b", clean))
    closers = len(re.findall(r"\bend\b", clean))
    repeats = len(re.findall(r"\brepeat\b", clean))
    expected_ends = openers - repeats

    if expected_ends > 0 and closers == 0:
        issues.append(f"Has {openers} block openers but no 'end' keywords")
        confidence -= 0.5
    elif expected_ends > 0 and abs(closers - expected_ends) > expected_ends * 0.5:
        issues.append("Block keyword imbalance")
        confidence -= 0.2

    if not re.search(r"\bfunction\b", clean):
        issues.append("No 'function' keyword found")
        confidence -= 0.2

    foreign_signals = [
        (r"\bdef\s+\w+\s*\(", "Python def"),
        (r"\bclass\s+\w+\s*[:{(]", "Class definition"),
        (r"\blet\s+\w+\s*=", "JS let"),
        (r"#include\s*<", "C/C++ include"),
        (r"\bfn\s+\w+\s*\(", "Rust fn"),
    ]
    for pattern, desc in foreign_signals:
        if re.search(pattern, clean, re.MULTILINE):
            issues.append(f"Foreign signal: {desc}")
            confidence -= 0.25

    luau_positives = [
        (r":\s*\w+\??\s*[,)=\n]", "type annotation"),
        (r"\blocal\s+function\b", "local function"),
        (r"\blocal\s+\w+\s*:", "typed local"),
        (r"--!strict", "strict mode"),
        (r"\bconst\s+\w+", "const declaration (Luau 711+)"),
        (r"\btable\.\w+\s*\(", "table stdlib"),
        (r"\bmath\.\w+\s*\(", "math stdlib"),
        (r"\bstring\.\w+\s*\(", "string stdlib"),
        (r"\btostring\s*\(|tonumber\s*\(", "type coercion"),
        (r"\bpcall\s*\(", "error handling"),
    ]
    bonus = 0.0
    for pattern, _ in luau_positives:
        if re.search(pattern, clean):
            bonus += 0.05
    confidence = min(1.0, confidence + min(bonus, 0.30))

    valid = confidence > 0.35 and len(issues) < 4
    return {"valid": valid, "confidence": max(0.0, confidence), "issues": issues}


def check_patterns(code: str, patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pat_str in patterns:
        try:
            if re.search(pat_str, code, re.MULTILINE):
                found.append(pat_str)
        except re.error:
            pass
    return found
