.PHONY: install install-dev install-torch lint format test figures tables benchmarks all clean report

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,experiments]"

install-torch:
	$(PIP) install -e ".[dev,experiments,torch]"

lint:
	$(PYTHON) -m ruff check src tests scripts

format:
	$(PYTHON) -m ruff format src tests scripts

test:
	$(PYTHON) -m pytest tests -v

figures:
	bash scripts/run_all.sh --figures

tables:
	bash scripts/run_all.sh --tables

benchmarks:
	bash scripts/run_all.sh --benchmarks

all:
	bash scripts/run_all.sh

# Rebuild docs/report/main.pdf. Requires pdfLaTeX (MacTeX/BasicTeX/TeX Live).
# Tectonic alone is insufficient because the SAIM class uses pdfTeX-only
# microtype features.
report:
	@command -v latexmk >/dev/null 2>&1 || { \
	  echo "ERROR: latexmk not found. Install BasicTeX with"; \
	  echo "         brew install --cask basictex"; \
	  echo "       and re-run 'make report'."; \
	  exit 2; }
	@bash scripts/run_all.sh --figures
	cd docs/report && latexmk -pdf main.tex

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
