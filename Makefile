# kv-census: deterministic research pipeline. See BRIEF.md / AGENTS.md.
SHELL := /bin/bash
VENV := .venv/bin/python
export CUDA_VISIBLE_DEVICES := ""
export HF_HUB_OFFLINE := "0"
export TOKENIZERS_PARALLELISM := "false"

.PHONY: data analysis validate paper test

data:
	$(VENV) scripts/fetch_census.py
	$(VENV) scripts/build_models_csv.py

analysis:
	$(VENV) analysis/run_analysis.py

validate:
	$(VENV) analysis/validate_experiment.py

paper:
	cd paper && TECTONIC_CACHE_DIR=../.tools/tectonic-cache ../.tools/tectonic -X compile main.tex

test:
	$(VENV) -m pytest tests/ -q
