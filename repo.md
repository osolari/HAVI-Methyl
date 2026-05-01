Read the entire report in docs/report/ — every section, appendix, figure caption, table caption, and algorithm block. Understand the mathematical framework, the algorithms, and every experimental claim before writing any code.

Then build out this repository end-to-end in the following order:

## 1. Project scaffolding

Set up:
- pyproject.toml with package metadata, dependencies (core: torch, numpy, scipy; dev: pytest, ruff, mypy; experiments: matplotlib, pandas, transformers, datasets), and tool configs
- Makefile with targets: install, install-dev, lint, format, test, figures, tables, benchmarks
- .gitignore, pre-commit config, Dockerfile
- src/<package_name>/__init__.py exporting the public API

## 2. Core library (src/)

Implement every algorithm, theorem, and formula described in the report as clean, composable Python modules. Each mathematical object in the report should map to a function or class:
- Density functions, distributions, and analytical formulas → dedicated modules
- Encoders, decoders, quantizers → separate modules with clear interfaces
- Codebook/dictionary construction (including any optimization or refinement) → own module
- Constants, closed-form expressions, asymptotic formulas → constants module
- Any variants or extensions (e.g., gain-corrected, weighted, task-adaptive) → own modules
- Utility functions (sampling, validation, metrics) → utils module

Design principles:
- Every public function has type hints and a one-line docstring referencing the theorem/equation number
- Numerical routines use float64 by default for precision
- All randomness flows through explicit generator/seed arguments for reproducibility
- Functions should be pure where possible — no hidden state

## 3. Test suite (tests/)

Write comprehensive tests that verify the theoretical claims in the report:
- For each theorem: a test that checks the formula empirically (Monte Carlo vs analytical)
- For each algorithm: correctness tests (valid output shapes, ranges, constraints)
- For each identity/property claimed: a statistical test (e.g., unbiasedness → empirical mean ≈ 0)
- Integration tests: full pipeline end-to-end
- Edge cases: boundary conditions, degenerate inputs, special cases mentioned in the report
- Use parametrized fixtures for dimensions, bit rates, and other swept parameters

All tests must pass. Run them and fix any failures before proceeding.

## 4. Figure scripts (scripts/fig*.py)

For EACH figure referenced in the report (\includegraphics or figure environment):
- Create a standalone script that generates the exact figure
- Read the caption carefully — it specifies the panels, parameters, datasets, and baselines
- Use publication-quality matplotlib styling (colorblind-safe palette, proper labels, LaTeX-rendered text)
- Save as both PDF and PNG to outputs/
- Auto-copy PDFs to docs/report/figures/
- Support --fast flag for quick iteration with fewer samples

Each script should be runnable independently and produce the complete figure.

## 5. Table scripts (scripts/tab*.py)

For EACH table in the report:
- If the table contains analytical/computed values: write a script that computes them from the library
- If the table contains experimental results: write a benchmark script (see below)
- If the table documents a protocol or design: write a script that formats the protocol as CSV
- Save outputs as CSV to outputs/ and copy to docs/report/tables/

## 6. Benchmark scripts (scripts/bench_*.py)

For EACH experimental table that requires running a model or heavy computation:
- Implement a complete benchmark script with real evaluation logic
- Use the core library for quantization/encoding
- For LLM benchmarks: load model, apply method, measure metrics (PPL, accuracy, latency)
- For retrieval benchmarks: build index, run queries, measure recall
- Support configurable parameters (model, bit rates, sample counts)
- Save results as CSV with columns matching the report table schema
- Competitor baselines that require external codebases: emit XX placeholder rows with a comment

Create shared utilities (_common.py, _style.py, etc.) for code reuse across scripts.

## 7. Orchestration

Create scripts/run_all.sh that runs tables → figures → benchmarks in order, with --fast and --benchmarks flags.

## 8. Notebooks

Create 2-3 tutorial notebooks demonstrating the library's core functionality.

## 9. README

Write a comprehensive README.md covering: what the project does, installation, quickstart, reproducing the paper, and API overview.

## Verification

After building everything:
1. Run the full test suite — all tests must pass
2. Run all figure scripts — verify PDFs are generated
3. Run all table scripts — verify CSVs are generated
4. Run benchmarks that don't require large GPU models
5. Verify zero XX placeholders exist in any generated table that should have computed values
6. Check that every \includegraphics and \input{tables/...} in the LaTeX has a corresponding generated file

Work systematically — implement, test, fix, then move to the next module. Do not write tests you haven't run or scripts you haven't executed.
