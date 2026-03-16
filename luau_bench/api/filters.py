from __future__ import annotations

import re

from luau_bench.api import register_filter


@register_filter("extract_code")
def extract_code(text: str, *, lang: str = "luau", **kwargs) -> str:
    pattern = re.compile(rf"```{re.escape(lang)}\s*\n(.*?)```", re.DOTALL)
    m = pattern.search(text)
    if m:
        return m.group(1).strip()

    if lang == "luau":
        m = re.search(r"```lua\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()

    m = re.search(r"```[a-z]*\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    code_pat = re.compile(
        r"^\s*(?:local\s|function\s|--!?|return\b|if\b|for\b|while\b|"
        r"repeat\b|type\s|export\s|end\b|\}|table\.|string\.|math\.)"
    )
    lines = text.split("\n")
    best: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        if code_pat.match(stripped) or (current and stripped):
            current.append(stripped)
        else:
            if len(current) > len(best):
                best = current[:]
            current = []
    if len(current) > len(best):
        best = current
    if len(best) >= 2:
        return "\n".join(best).strip()

    return text.strip()


@register_filter("strip_whitespace")
def strip_whitespace(text: str, **kwargs) -> str:
    return text.strip()


@register_filter("lowercase")
def lowercase(text: str, **kwargs) -> str:
    return text.lower()


@register_filter("first_line")
def first_line(text: str, **kwargs) -> str:
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


@register_filter("regex_extract")
def regex_extract(text: str, *, pattern: str = "", group: int = 0, **kwargs) -> str:
    if not pattern:
        return text
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return m.group(group)
    return text


@register_filter("remove_comments")
def remove_comments(text: str, **kwargs) -> str:
    return re.sub(r"--[^\n]*", "", text)


@register_filter("truncate")
def truncate(text: str, *, max_chars: int = 10000, **kwargs) -> str:
    return text[:max_chars]
