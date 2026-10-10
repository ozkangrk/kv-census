# Shared interfaces (written by the orchestrator in Phase 0; do not edit without orchestrator approval)

All code runs with `CUDA_VISIBLE_DEVICES=""` and the project venv `.venv/bin/python`.

## 1. `analysis/kv.py` — analytic KV accounting (owner: kv-accounting)

Pure functions over a raw HF `config.json` **dict** (no network, no HF library imports for parsing;
the caller supplies the dict). Multimodal configs nest the text backbone under
`text_config` / `llm_config` / `language_config` / `text_model`; use a helper that picks the
right scope.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LayerKV:
    kind: str                  # "full" | "window" | "latent" | "recurrent"
    subtype: str               # "mha"|"gqa"|"mqa"|"swa"|"local"|"mla"|"mamba2"|"mamba"|"linear"|"gdn"|"deltanet"|"rwkv"|...
    n_kv_heads: int            # 0 when not applicable (latent/recurrent)
    head_dim: int              # 0 when not applicable
    bytes_per_element: int     # 2 (BF16) by default
    window: int | None         # max cached tokens for "window" layers, None otherwise
    latent_elems: int | None   # per-token latent elements for "latent" (MLA): kv_lora_rank + qk_rope_head_dim
    fixed_state_bytes: float   # constant per-layer state bytes (recurrent layers), else 0.0

def text_scope(config: dict) -> dict
    # Returns the language-backbone sub-dict of a (possibly multimodal) config.

def parse_layers(config: dict, *, bytes_per_element: int = 2) -> list[LayerKV]
    # One entry per cache-owning unit. Cross-layer shared KV (e.g. YOCO/CrossKV-style
    # or layer-groups sharing one KV) is emitted ONCE; document how sharing was detected
    # from the config in the module docstring.

def kv_bytes_per_token(config: dict, *, bytes_per_element: int = 2) -> float
    # Asymptotic marginal bytes per extra token as T -> infinity: only layers whose
    # cache grows without bound ("full" and "latent"). Window layers have marginal 0,
    # recurrent layers 0. For a pure mha/gqa/mqa model this equals
    # 2 * num_hidden_layers * num_key_value_heads(or num_attention_heads) * head_dim * bytes.

def kv_bytes_at(config: dict, T: int, *, bytes_per_element: int = 2) -> float
    # Total cache + state bytes for ONE sequence at context length T (theoretical minimum):
    #   full:   2*kvh*hd*b*T
    #   window: 2*kvh*hd*b*min(T, W)
    #   latent: latent_elems*b*T          (theoretical latent; as-implemented goes in validation)
    #   recurrent: fixed_state_bytes (independent of T)
    # Shared caches counted once.

def fixed_state_bytes(config: dict, *, bytes_per_element: int = 2) -> float
    # Sum of constant per-layer state bytes (recurrent layers only).

def kv_par_context(config: dict, *, total_params: float, weight_bytes_per_param: float = 2.0,
                   bytes_per_element: int = 2) -> float
    # Smallest T with kv_bytes_at(T) >= total_params * weight_bytes_per_param.
    # For models with only bounded caches (pure window/recurrent), return the T where the
    # bounded cache saturates if it is below parity, else math.inf; document the convention.
```

Rules:
- Unknown architecture => raise `KVConfigError` (define it) rather than silently guessing;
  the census builder records such rows and the orchestrator decides.
- Head-dim convention: explicit `head_dim` wins; else `hidden_size // num_attention_heads`.
- Unit tests in `tests/test_kv.py` use hand-written synthetic config dicts (no network),
  including at least: pure MHA, pure GQA, MQA, GQA+sliding-window (e.g. Mistral-style
  `sliding_window` with per-layer pattern or all-window), MLA (deepseek-v2 style
  `kv_lora_rank`/`qk_rope_head_dim`), hybrid recurrent (granite/falcon-h1 style layer
  pattern strings), Gemma-style `head_dim > hidden/heads`, cross-layer shared KV.
- `make test` must pass before Phase 1 acceptance.

## 2. `data/models.csv` schema (owner: census-data)

One row per included checkpoint, UTF-8, comma-separated, header row with EXACTLY these columns:

repo_id, org, model_name, release_date, params_source, total_params_b, active_params_b, moe,
num_layers, num_attention_heads, num_kv_heads, head_dim, hidden_size, attention_type,
gqa_ratio, window_size, n_full_layers, n_window_layers, n_latent_layers, n_recurrent_layers,
kv_shared_cross_layer, kv_lora_rank, qk_rope_head_dim, recurrent_state_bytes,
kv_bytes_per_token_bf16, kv_bytes_at_32k, kv_bytes_at_128k, dedup_group, notes

- `repo_id`: `org/name`, unique, and `data/raw/configs/<org>__<name>.json` must exist for every row.
- `release_date`: `YYYY-MM-DD` from the Hub `createdAt`.
- `total_params_b`: billions; `params_source` is `safetensors` (Hub safetensors metadata)
  or `config` (computed from the config; formula recorded in the script).
- `attention_type`: `mha`/`gqa`/`mqa` ONLY for pure full-attention models; else a
  descriptive label (`gqa+swa`, `swa`, `local+global`, `hybrid-mamba`, `mla`, `mla+swa`, ...).
- `gqa_ratio`: `num_attention_heads / num_kv_heads` where applicable, else empty.
- `kv_bytes_per_token_bf16` etc. computed by importing `analysis/kv.py` (never re-implement).
- `dedup_group`: stable id (e.g. `sha1(org + base name stem + geometry signature)[:12]`)
  grouping base/instruct twins with identical geometry; trend statistics dedup on this.
- Empty numeric cells stay empty (never 0 unless truly 0).

Side outputs of the census builder:
- `data/gated_skipped.txt`: one repo_id per line, counted as selection-bias limitation.
- `data/raw/hub_api/`: cached raw JSON API responses (model info, safetensors metadata).
- `data/raw/configs/`: one config.json per included repo, named `<org>__<name>.json`.

## 3. `results/kv_validation.csv` schema (owner: validation)

model_or_arch, attention_class, analytic_bytes, measured_bytes, rel_error, explanation

- `rel_error = abs(measured - analytic) / analytic`, recomputed by the gate to 1e-3.
- >= 6 rows, >= 4 distinct attention_class values, >= 5 rows within 1% (others need an
  `explanation`, which is a finding, not an apology).

## 4. `paper/generated/*.tex` (owner: analysis phase 2)

Macros `\newcommand{\kvc<Name>}{<value>}` and full `tabular`/`table` environments.
Every number in the paper comes from here via `\input{generated/...}`. Deterministic:
sorted rows, fixed seeds (0), no timestamps; figures are PDF with fixed metadata
(`SOURCE_DATE_EPOCH=0` or explicit `metadata={'CreationDate': None}`).

## 5. File ownership map

- orchestrator: Makefile, INTERFACES.md, PLAN.md, analysis/run_analysis.py (Phase 2),
  paper/main.tex integration (Phase 3), PROGRESS.md.
- census-data: scripts/fetch_census.py, scripts/build_models_csv.py, data/raw/**
  (except data/raw/literature/), data/models.csv, data/gated_skipped.txt.
- kv-accounting: analysis/kv.py, tests/test_kv.py.
- validation: analysis/validate_experiment.py, results/kv_validation.csv.
- literature: paper/references.bib, data/raw/literature/**, notes/related_work.md.
