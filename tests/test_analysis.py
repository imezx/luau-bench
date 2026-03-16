from luau_bench.analysis.static import analyze, check_code_validity, check_patterns


class TestAnalyze:
    SAMPLE = """\
--!strict

type Point = { x: number, y: number }

local function distance(a: Point, b: Point): number
    local dx = a.x - b.x
    local dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)
end

local function midpoint(a: Point, b: Point): Point
    return {
        x = (a.x + b.x) / 2,
        y = (a.y + b.y) / 2,
    }
end
"""

    def test_line_counts(self):
        a = analyze(self.SAMPLE)
        assert a.total_lines > 0
        assert a.code_lines > 0

    def test_strict_mode(self):
        a = analyze(self.SAMPLE)
        assert a.has_strict_mode is True

    def test_type_alias(self):
        a = analyze(self.SAMPLE)
        assert a.type_aliases >= 1

    def test_functions(self):
        a = analyze(self.SAMPLE)
        assert a.function_count >= 2
        assert a.local_function_count >= 2

    def test_typed_params(self):
        a = analyze(self.SAMPLE)
        assert a.typed_params > 0

    def test_api_calls(self):
        a = analyze(self.SAMPLE)
        assert "math.sqrt" in a.api_calls

    def test_type_coverage(self):
        a = analyze(self.SAMPLE)
        assert a.type_coverage > 50.0

    def test_empty_code(self):
        a = analyze("")
        assert a.total_lines == 1
        assert a.code_lines == 0

    def test_modern_features(self):
        code = """\
local x += 1
local s = `hello {name}`
local y = if cond then 1 else 2
"""
        a = analyze(code)
        assert a.uses_compound_assignment is True
        assert a.uses_string_interpolation is True
        assert a.uses_if_expression is True


class TestCodeValidity:
    def test_valid_luau(self):
        code = "local function foo()\n    return 1\nend"
        result = check_code_validity(code)
        assert result["valid"] is True

    def test_empty_code(self):
        result = check_code_validity("")
        assert result["valid"] is False

    def test_python_code(self):
        code = "def foo():\n    return 1\n"
        result = check_code_validity(code)
        assert result["confidence"] < 0.7

    def test_javascript(self):
        code = "const x = 1;\nlet y = 2;\nfunction foo() { return x + y; }"
        result = check_code_validity(code)
        assert result["confidence"] < 0.7


class TestCheckPatterns:
    def test_finds_pattern(self):
        code = "local x = table.sort(t)"
        found = check_patterns(code, [r"table\.sort"])
        assert r"table\.sort" in found

    def test_misses_pattern(self):
        code = "local x = 1"
        found = check_patterns(code, [r"table\.sort"])
        assert found == []

    def test_invalid_regex(self):
        found = check_patterns("test", ["[invalid"])
        assert found == []
