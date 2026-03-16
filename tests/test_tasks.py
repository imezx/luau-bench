import yaml

from luau_bench.api.task import ConfigurableTask, parse_task_config
from luau_bench.tasks import load_tasks_from_path


class TestParseTaskConfig:
    def test_minimal(self):
        raw = {
            "task": "my_task",
            "doc_to_text": "{{ question }}",
            "metric_list": [{"metric": "exact_match"}],
            "docs": [{"question": "What?", "answer": "That."}],
        }
        config = parse_task_config(raw)
        assert config.task == "my_task"
        assert len(config.metric_list) == 1
        assert config.metric_list[0].metric == "exact_match"
        assert len(config.docs) == 1

    def test_full_config(self):
        raw = {
            "task": "full_task",
            "group": "test_group",
            "version": 2.0,
            "system_prompt": "Be helpful.",
            "doc_to_text": "{{ q }}",
            "doc_to_target": "{{ a }}",
            "generation_kwargs": {
                "max_tokens": 512,
                "temperature": 0.5,
                "stop_sequences": ["END"],
            },
            "filters": [
                {"name": "strip_whitespace"},
                {"name": "extract_code", "args": {"lang": "lua"}},
            ],
            "metric_list": [
                {"metric": "exact_match", "primary": True},
                {"metric": "contains", "args": {"ignore_case": True}},
            ],
            "metadata": {"author": "test"},
            "docs": [{"q": "hi", "a": "hello"}],
        }
        config = parse_task_config(raw)
        assert config.task == "full_task"
        assert config.group == "test_group"
        assert config.version == 2.0
        assert config.generation_kwargs.max_tokens == 512
        assert config.generation_kwargs.temperature == 0.5
        assert len(config.filters) == 2
        assert config.filters[1].args["lang"] == "lua"
        assert len(config.metric_list) == 2
        assert config.metric_list[0].primary is True


class TestConfigurableTask:
    def _make_task(self, **overrides):
        raw = {
            "task": "test",
            "system_prompt": "You are helpful.",
            "doc_to_text": "Question: {{ question }}",
            "doc_to_target": "{{ answer }}",
            "metric_list": [{"metric": "exact_match"}],
            "docs": [
                {"question": "What is 1+1?", "answer": "2"},
                {"question": "What is 2+2?", "answer": "4"},
            ],
        }
        raw.update(overrides)
        config = parse_task_config(raw)
        return ConfigurableTask(config)

    def test_get_docs(self):
        task = self._make_task()
        docs = task.get_docs()
        assert len(docs) == 2
        assert docs[0]["question"] == "What is 1+1?"

    def test_build_prompt(self):
        task = self._make_task()
        prompt = task.build_prompt({"question": "Hi?"})
        assert prompt["system"] == "You are helpful."
        assert prompt["user"] == "Question: Hi?"

    def test_get_target(self):
        task = self._make_task()
        target = task.get_target({"answer": "42"})
        assert target == "42"

    def test_get_metrics(self):
        task = self._make_task()
        specs = task.get_metric_specs()
        assert len(specs) == 1
        assert specs[0].metric == "exact_match"

    def test_empty_system_prompt(self):
        task = self._make_task(system_prompt="")
        prompt = task.build_prompt({"question": "Hi?"})
        assert prompt["system"] == ""

    def test_jinja_loop(self):
        task = self._make_task(
            doc_to_text="{% for ex in examples %}{{ ex }} {% endfor %}",
        )
        prompt = task.build_prompt({"examples": ["a", "b", "c"]})
        assert "a" in prompt["user"]
        assert "c" in prompt["user"]


class TestLoader:
    def test_load_single_task(self, tmp_path):
        task_yaml = {
            "task": "loaded_task",
            "doc_to_text": "{{ q }}",
            "metric_list": [{"metric": "exact_match"}],
            "docs": [{"q": "test"}],
        }
        (tmp_path / "task.yaml").write_text(yaml.dump(task_yaml))
        tasks = load_tasks_from_path(tmp_path)
        assert len(tasks) == 1
        assert tasks[0].config.task == "loaded_task"

    def test_load_multiple_tasks(self, tmp_path):
        for i in range(3):
            task_yaml = {
                "task": f"task_{i}",
                "doc_to_text": "{{ q }}",
                "metric_list": [{"metric": "exact_match"}],
                "docs": [{"q": "test"}],
            }
            (tmp_path / f"task_{i}.yaml").write_text(yaml.dump(task_yaml))
        tasks = load_tasks_from_path(tmp_path)
        assert len(tasks) == 3

    def test_load_group(self, tmp_path):
        task_yaml = {
            "task": "a_task",
            "doc_to_text": "{{ q }}",
            "metric_list": [{"metric": "exact_match"}],
            "docs": [{"q": "test"}],
        }
        (tmp_path / "task.yaml").write_text(yaml.dump(task_yaml))

        group_yaml = {
            "group": "my_group",
            "task": ["a_task"],
        }
        (tmp_path / "group.yaml").write_text(yaml.dump(group_yaml))

        tasks = load_tasks_from_path(tmp_path)
        assert len(tasks) == 1

        from luau_bench.api import list_groups

        groups = list_groups()
        assert "my_group" in groups

    def test_load_nonexistent_path(self, tmp_path):
        tasks = load_tasks_from_path(tmp_path / "nonexistent")
        assert tasks == []

    def test_duplicate_task_skipped(self, tmp_path):
        task_yaml = {
            "task": "dup",
            "doc_to_text": "{{ q }}",
            "metric_list": [{"metric": "exact_match"}],
            "docs": [{"q": "test"}],
        }
        (tmp_path / "a.yaml").write_text(yaml.dump(task_yaml))
        (tmp_path / "b.yaml").write_text(yaml.dump(task_yaml))
        tasks = load_tasks_from_path(tmp_path)
        assert len(tasks) == 1

    def test_include_inheritance(self, tmp_path):
        base = {
            "task": "base_task",
            "system_prompt": "Be helpful.",
            "doc_to_text": "{{ q }}",
            "metric_list": [{"metric": "exact_match"}],
            "docs": [{"q": "base"}],
        }
        (tmp_path / "base.yaml").write_text(yaml.dump(base))

        child = {
            "include": "base.yaml",
            "task": "child_task",
            "doc_to_text": "Override: {{ q }}",
        }
        (tmp_path / "child.yaml").write_text(yaml.dump(child))

        tasks = load_tasks_from_path(tmp_path)
        names = {t.config.task for t in tasks}
        assert "child_task" in names

        child_task = next(t for t in tasks if t.config.task == "child_task")
        assert child_task.config.system_prompt == "Be helpful."
        prompt = child_task.build_prompt({"q": "hi"})
        assert prompt["user"] == "Override: hi"
