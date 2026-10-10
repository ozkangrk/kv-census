# İlerleme Günlüğü (PROGRESS.md)

## Phase 0 — Kurulum (tamamlandı)
- Depo hazırdı (commit 8418bbf: BRIEF.md + AGENTS.md). İskelet dizinler, Makefile
  (data/analysis/validate/paper/test hedefleri), `analysis/INTERFACES.md` (kv.py API'si,
  models.csv şeması, dosya sahiplik haritası) ve ön-kayıtlı `analysis/PLAN.md` yazıldı.
- `uv` ile `.venv` (Python 3.12) kuruldu; torch CPU-only, transformers, pandas, numpy,
  scipy, matplotlib, requests, pytest; sürümler `requirements.txt`'e sabitlendi.
- Sonraki adım: Phase 1 — 4 paralel alt-ajan (census-data, kv-accounting, validation,
  literature).
