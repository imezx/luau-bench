import pytest

from luau_bench.api import (
    get_filter,
    get_metric,
    list_filters,
    list_metrics,
    register_group,
    resolve_tasks,
)

import luau_bench.api.metrics
import luau_bench.api.filters


class TestMetricRegistry:
    def test_builtin_metrics_registered(self):
        names = list_metrics()
        assert "exact_match" in names
        assert "contains" in names
        assert "regex_match" in names
        assert "pass_at_k" in names
        assert "luau_exec" in names
        assert "luau_static_analysis" in names
        assert "luau_analyze" in names

    def test_get_known_metric(self):
        fn = get_metric("exact_match")
        assert callable(fn)

    def test_get_unknown_metric_raises(self):
        with pytest.raises(KeyError, match="Unknown metric"):
            get_metric("nonexistent_metric_xyz")

    def test_exact_match(self):
        fn = get_metric("exact_match")
        result = fn(["a", "b", "c"], ["a", "b", "x"])
        assert result["exact_match"] == pytest.approx(2 / 3 * 100.0)

    def test_exact_match_ignore_case(self):
        fn = get_metric("exact_match")
        result = fn(["Hello"], ["hello"], ignore_case=True)
        assert result["exact_match"] == pytest.approx(100.0)

    def test_contains(self):
        fn = get_metric("contains")
        result = fn(["hello world", "foo"], ["world", "bar"])
        assert result["contains"] == pytest.approx(50.0)

    def test_regex_match(self):
        fn = get_metric("regex_match")
        result = fn(["abc123", "xyz"], ["\\d+", "\\d+"])
        assert result["regex_match"] == pytest.approx(50.0)


class TestFilterRegistry:
    def test_builtin_filters_registered(self):
        names = list_filters()
        assert "extract_code" in names
        assert "strip_whitespace" in names
        assert "lowercase" in names
        assert "first_line" in names
        assert "regex_extract" in names

    def test_get_known_filter(self):
        fn = get_filter("strip_whitespace")
        assert callable(fn)

    def test_get_unknown_filter_raises(self):
        with pytest.raises(KeyError, match="Unknown filter"):
            get_filter("nonexistent_filter_xyz")

    def test_extract_code_luau(self):
        fn = get_filter("extract_code")
        text = "Here is code:\n```luau\nlocal x = 1\n```\nDone."
        assert fn(text) == "local x = 1"

    def test_extract_code_lua_fallback(self):
        fn = get_filter("extract_code")
        text = "```lua\nprint('hi')\n```"
        assert fn(text) == "print('hi')"

    def test_extract_code_any_block_fallback(self):
        fn = get_filter("extract_code")
        text = "```\nlocal y = 2\n```"
        assert fn(text) == "local y = 2"

    def test_first_line(self):
        fn = get_filter("first_line")
        assert fn("\n  hello\nworld\n") == "hello"

    def test_lowercase(self):
        fn = get_filter("lowercase")
        assert fn("Hello World") == "hello world"

    def test_truncate(self):
        fn = get_filter("truncate")
        assert fn("abcdef", max_chars=3) == "abc"

    def test_regex_extract(self):
        fn = get_filter("regex_extract")
        result = fn("The answer is 42.", pattern=r"(\d+)", group=1)
        assert result == "42"


class TestGroups:
    def test_resolve_individual_tasks(self):
        assert resolve_tasks(["a", "b"]) == ["a", "b"]

    def test_resolve_group(self):
        register_group("my_group", ["t1", "t2", "t3"])
        result = resolve_tasks(["my_group"])
        assert result == ["t1", "t2", "t3"]

    def test_resolve_mixed(self):
        register_group("g", ["t1", "t2"])
        result = resolve_tasks(["g", "t3"])
        assert result == ["t1", "t2", "t3"]

    def test_resolve_deduplicates(self):
        register_group("g2", ["t1", "t2"])
        result = resolve_tasks(["t1", "g2"])
        assert result == ["t1", "t2"]
