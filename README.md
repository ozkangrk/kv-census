# Memory Is Architecture: a KV-cache census of open-weight LLMs (2023–2026)

Data, pipeline, and paper for *"Memory Is Architecture: A Census of KV-Cache Footprints in
Open-Weight Language Models, 2023–2026"* (Gurkan Ozkan, Istanbul Technical University).

**AI-use disclosure:** this study was produced end to end by AI systems. An AI assistant
(Claude, Anthropic) proposed the question and wrote the brief (`BRIEF.md`), the agent rules
(`AGENTS.md`), and an external completion gate. An autonomous agent swarm (Prime Agent, GLM-5.3)
executed the census, the analysis, and the drafting. The human author reviewed the manuscript and
takes responsibility for it. See the paper's AI-Use Disclosure section. `PROGRESS.md` and
`SUMMARY_TR.md` are the swarm's own progress log and summary (in Turkish).

## Layout
| Path | Contents |
|---|---|
| `data/models.csv` | The census: one row per checkpoint (2,331 rows) |
| `data/raw/` | Snapshotted `config.json` files and cached Hugging Face Hub / literature API responses (so re-runs are offline) |
| `analysis/kv.py` | Analytic KV accounting for every attention class (full/GQA/MQA, sliding window, MLA, Mamba/hybrid, cross-layer sharing) |
| `analysis/PLAN.md` | Pre-registered analysis plan, with dated deviations |
| `analysis/validate_experiment.py` | CPU validation: analytic vs. measured HF `transformers` cache bytes |
| `results/` | Analysis outputs |
| `paper/` | LaTeX source. Every number comes from `paper/generated/`, which `make analysis` writes |
| `reviews/` | Adversarial review reports and responses |

## Reproduce
```bash
uv venv && uv pip install -r requirements.txt
make test        # unit tests for kv.py
make analysis    # offline, deterministic: regenerates results/, paper/figures, paper/generated
make validate    # CPU-only validation experiment (results/kv_validation.csv)
make paper       # builds paper/main.pdf with tectonic (on PATH, or .tools/tectonic)
make data        # optional: re-fetch from the Hugging Face Hub (network)
```
