# Luau Bench

An evaluation harness for benchmarking LLM-generated Luau code. Highly-inspired from `EleutherAI/lm-evaluation-harness`.

## Installation

```bash
git clone https://github.com/imezx/luau-bench.git
cd luau-bench

# With all optional providers & datasets
pip install -e ".[all]"

# For specific providers
pip install -e ".[anthropic]"  # Claude models
pip install -e ".[datasets]"   # HuggingFace dataset loader
pip install -e ".[progress]"   # tqdm progress bars for long runs

# through pip
pip install luau-bench --no-cache-dir
```

## Runtime Setup

Luau Bench requires a Luau runtime for executing generated code. Install one or more with the bundled script:

```bash
# Install everything (Luau CLI + Zune + StyLua)
./tools/install-runtime.sh --all

# Install Zune only (recommended runtime)
./tools/install-runtime.sh --zune

# Install the Luau CLI (also installs luau-analyze for static diagnostics)
./tools/install-runtime.sh --luau

# verify the environment
luau-bench info
```

Supported runtimes in priority order: `luau`, `zune`. The `LUAU_RUNTIME` environment variable can pin a specific binary path.

## Quick Start

```bash
mkdir my_tasks
cp templates/task_basic.yaml my_tasks/

luau-bench validate --include-path ./my_tasks

luau-bench selftest --include-path ./my_tasks

# Run a benchmark - results printed to terminal only
luau-bench run --provider ollama --model codellama \
    --include-path ./my_tasks

# Run & export results (all formats)
luau-bench run --provider ollama --model codellama \
    --include-path ./my_tasks --export

# Run & export specific formats
luau-bench run --provider ollama --model codellama \
    --include-path ./my_tasks --export html,json
```

## CLI Reference

```
luau-bench run          Evaluate a model against tasks
luau-bench ls           List available tasks & groups
luau-bench validate     Check task YAML files for errors
luau-bench selftest     Run reference solutions against test harnesses
luau-bench compare      Compare two or more result JSON files
luau-bench cache        Manage the generation result cache
luau-bench info         Show environment info
```

### `run`

```bash
luau-bench run \
    --provider ollama \
    --model codellama \
    --include-path ./my_tasks \
    --tasks task1,task2 \
    --num-samples 5 \
    --parallel 4 \
    --task-parallel 4 \
    --export html,json,md \
    --output ./results \
    --log-samples --show-samples \
    --temperature 0.0 \
    --max-tokens 4096 \
    -v
```

Results are printed to the terminal by default. Pass `--export` to write files. Use `--export` alone to export all formats (json, md, html), or `--export html,json` to select specific ones. The `--output` directory is only used when `--export` is active.

## Supported Providers

| Provider         | `--provider` value | Notes                                             |
| ---------------- | ------------------ | ------------------------------------------------- |
| vLLM             | `vllm`             | Full support including loglikelihood              |
| OpenAI           | `openai`           | Set `OPENAI_API_KEY`                              |
| Anthropic Claude | `anthropic`        | Set `ANTHROPIC_API_KEY`; no loglikelihood support |
| Ollama           | `ollama`           | Local models via `ollama serve`                   |
| TGI              | `tgi`              | HuggingFace Text Generation Inference             |
| LM Studio        | `lmstudio`         | Local via LM Studio server                        |

Loglikelihood scoring (`output_type: loglikelihood`) requires a backend that exposes the `/v1/completions` endpoint with `echo=True` & `logprobs=1`. This includes vLLM, TGI, & llama.cpp. It is **not** supported by the Anthropic adapter.

## Contributing

We accept any well-formed task that has a clear purpose, correct reference solution, & sufficient test coverage.
