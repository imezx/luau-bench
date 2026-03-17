# luau_coding

`HumanEval`-style coding benchmark for Luau. Each task provides a typed function
signature and docstring; the model must complete the implementation. Solutions
are evaluated by executing a test harness against the generated code.

## Coverage

| ID     | Difficulty | Category | Description                                    |
|--------|-----------|----------|------------------------------------------------|
| lc001  | easy      | string   | Count vowels in a string                       |
| lc002  | easy      | table    | Sum an array of numbers                        |
| lc003  | easy      | string   | Check if a string is a palindrome              |
| lc004  | easy      | math     | Compute factorial (with error on negative)     |
| lc005  | easy      | table    | Generic `filter` higher-order function         |
| lc006  | medium    | table    | Remove duplicate values preserving order       |
| lc007  | medium    | string   | Split a string by a plain delimiter            |
| lc008  | medium    | math     | nth Fibonacci number (iterative, n≤30)         |
| lc009  | medium    | table    | One-level flatten of a nested array            |
| lc010  | medium    | oop      | Generic Stack class (push/pop/peek/size)       |
| lc011  | medium    | string   | Normalise runs of whitespace                   |
| lc012  | medium    | math     | Primality test                                 |
| lc013  | hard      | table    | Recursive deep copy                            |
| lc014  | hard      | oop      | Generic Queue (O(1) amortised enqueue/dequeue) |
| lc015  | hard      | string   | `{key}` template string interpolation          |
| lc016  | hard      | table    | Invert a table (swap keys and values)          |
| lc017  | hard      | types    | `memoize` higher-order function                |
| lc018  | hard      | types    | `pipe` higher-order function                   |
| lc019  | hard      | roblox   | Signal/connection system (Roblox-style)        |
| lc020  | hard      | roblox   | Hierarchical state machine with enter/exit     |

## Metrics

- **luau_exec** (primary) — pass rate against the bundled test harnesses.
  Requires a Luau runtime (`zune` or `luau`).
- **luau_static_analysis** — measures type coverage, strict-mode usage,
  locality ratio, and code validity.

## Few-shot / stratified sampling

This task supports stratified few-shot sampling by difficulty:

```yaml
num_fewshot: 3
fewshot_strategy: stratified
fewshot_stratify_field: difficulty
```

This ensures shots are sampled roughly equally from easy/medium/hard rather
than being drawn randomly (which may over-sample any one difficulty tier).

## Adding more tasks

Each JSONL line must have:
- `id` — unique string identifier
- `difficulty` — `easy`, `medium`, or `hard`
- `category` — e.g. `string`, `math`, `table`, `oop`, `types`, `roblox`
- `description` — the Luau function signature + docstring shown to the model
- `reference_solution` — the correct Luau code (used by `selftest`)
- `test_harness` — Luau script with `{{CODE}}` placeholder; outputs
  `@@LUAU_BENCH_RESULT@@PASS:label` / `@@LUAU_BENCH_RESULT@@FAIL:label:msg`
