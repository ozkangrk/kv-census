# Research brief: KV-cache census of open-weight LLMs (2023–2026)

## Working title
**Memory Is Architecture: A Census of KV-Cache Footprints in Open-Weight Language Models, 2023–2026**

## Motivation
Practitioners size serving hardware by parameter count. For long-context and
multi-sequence serving, the binding constraint is usually the KV cache. Its size
is set by attention geometry: layers × KV heads × head dim. It is also set by
attention scope (global vs. sliding-window/local) and by state type (growing KV
vs. the fixed recurrent state of Mamba/linear-attention layers, or MLA latents).
Parameter count does not set it. On unified-memory edge devices (for example,
a 128 GB GB10 / DGX Spark), this decides how many concurrent long contexts fit.

Anecdotal motivating observation (UNVERIFIED; recompute it, never quote it). At 4
sequences × 32k tokens, BF16 KV:
- Qwen3-8B: about 24 GB
- Qwen3-30B-A3B: about 16 GB
- a Gemma-family 12B model with head_dim 256: about 64 GB

The larger model can need less KV.

## Research questions and hypotheses
- **RQ1 (trend):** How have KV bytes/token, and KV bytes/token per billion
  parameters, evolved for open-weight decoder LLMs released 2023–2026?
  *H1: the median per-parameter KV cost fell substantially.*
- **RQ2 (mechanisms):** How much cross-model variance is explained by:
  - GQA ratio
  - head_dim
  - depth
  - local/sliding-window layers
  - hybrid recurrent layers
  - latent compression (MLA)
  - cross-layer KV sharing

  *H2: parameter count is a weak predictor of KV footprint (low log-log R²) compared with attention geometry.*
  *H3: MoE models have much lower KV per total parameter.*
- **RQ3 (capacity):** Assume fixed memory budgets (24 GB, 80 GB, 128 GB unified)
  and weight precisions (BF16, FP8, 4-bit), with KV in BF16 and FP8. What context
  × concurrency can each model serve? Define the **KV-parity context**: the
  context length at which one sequence's KV bytes equal the weight bytes. Study
  how it trended. *H4: KV-parity context grew by orders of magnitude.
  Sliding-window and hybrid designs give the largest savings, but the savings
  depend on context length.*
- **RQ4 (validity):** Do analytic estimates match the cache tensors that a
  reference implementation (HF transformers) actually allocates? Where do
  implementations diverge from the theoretical minimum? Examples: MLA cached
  decompressed, sliding windows allocated in full.

Report what the data says, including null or contrary results. A refuted
hypothesis is a valid finding.

## Data: inclusion criteria
**Include:**
- Open-weight, decoder-only text LLMs on the Hugging Face Hub, released from
  2023-01 to today.
- For multimodal models, use the language backbone.
- Only original publisher orgs. Examples: meta-llama, mistralai, Qwen, google,
  microsoft, deepseek-ai, 01-ai, tiiuae, allenai, nvidia, ibm-granite,
  CohereLabs/CohereForAI, zai-org/THUDM, moonshotai, MiniMaxAI, openai,
  internlm, baichuan-inc, stabilityai, HuggingFaceTB, apple, LiquidAI,
  ServiceNow-AI, ai21labs, and others you can justify.

**Exclude:**
- Fine-tunes, merges, and quantized re-uploads from other accounts.
- Gated repos whose config.json is not publicly downloadable. Count the
  skipped repos and report the count as a selection-bias limitation.
- Never use an HF token.

**Data requirements:**
- Snapshot every config.json to `data/raw/configs/<org>__<name>.json`. Cache all
  API responses under `data/raw/`.
- Take parameter counts from Hub safetensors metadata where available. Otherwise
  compute them from the config, and record the source in a column.
- Use the Hub `createdAt` timestamp as the release-date proxy, and state this as
  a limitation.
- Keep ≥ 150 checkpoints. Add a `dedup_group` column so that base/instruct
  twins with identical geometry count once in trend statistics.

## KV accounting conventions (implement in `analysis/kv.py` with unit tests)
- **Full attention layer:** `2 × n_kv_heads × head_dim × bytes` per token.
- **Sliding-window/local layer with window W:** at context T, it holds
  `min(T, W)` tokens.
- **MLA:** the theoretical latent is `(kv_lora_rank + qk_rope_head_dim) × bytes`
  per token. Record the as-implemented size separately.
- **Recurrent (Mamba/SSM, linear attention, DeltaNet, etc.) layer:** a constant
  state that does not depend on T. Compute it from the config.
- **Cross-layer KV sharing:** count each shared cache once.
- **`models.csv` columns:**
  - `kv_bytes_per_token_bf16`: the asymptotic marginal bytes per extra token as
    T → ∞ (global layers only), in BF16. For pure mha/gqa/mqa rows this must
    equal `2 × num_layers × num_kv_heads × head_dim × 2`.
  - `kv_bytes_at_32k` and `kv_bytes_at_128k`.
  - `attention_type`: label honestly. Use `mha`, `gqa`, or `mqa` only for pure
    full-attention models. Anything else (sliding/hybrid/MLA/shared) gets a
    descriptive label such as `gqa+swa`, `hybrid-mamba`, or `mla`.

## Capacity model (RQ3)
- Usable KV pool = budget − weight bytes − runtime overhead.
- Overhead is a parameter. Report a sensitivity analysis over several values.
- Weight bytes come from params × bytes/param. For 4-bit, model the scale
  overhead explicitly.
- Report:
  - max total tokens
  - max concurrent sequences at 32k and 128k
  - KV-parity context
  - the "KV-dominant" regime boundary

## Validation experiment (RQ4): CPU only, `CUDA_VISIBLE_DEVICES=""`
**Setup:**
- Instantiate reduced-depth, randomly initialised models from the real
  snapshotted configs using HF transformers (`from_config`, no weight downloads).
- Keep the per-layer attention geometry. Set `num_hidden_layers` to the smallest
  value that preserves one full period of the layer-type pattern. Shrink the
  vocabulary if needed.

**Run:**
- Prefill T tokens and measure the bytes of every cache tensor (keys, values,
  conv/ssm states).
- Compare with `analysis/kv.py` evaluated on the SAME reduced config.

**Coverage:**
- At least 6 models spanning at least 4 attention classes: mha/gqa,
  sliding-window or local/global, hybrid recurrent, and MLA.
- For windowed models, run T > W. A documented, reduced W is fine.

**Explaining gaps:** if measured size ≠ analytic (> 1%), explain why in the
`explanation` column. That gap is a finding, not a failure.

**Output:** `results/kv_validation.csv` with columns
`model_or_arch, attention_class, analytic_bytes, measured_bytes, rel_error, explanation`.

**Runtime:** must finish in < 10 min.

## Deliverables (the completion gate re-checks all of this)
**Makefile targets:**
- `make data`: network fetch from cached/raw snapshots. The gate does not run it.
- `make analysis`: OFFLINE and deterministic. Reads only `data/`, writes
  `results/`, `paper/figures/`, and `paper/generated/*.tex`. Every number in the
  paper must come from `paper/generated/` macros/tables via `\input`. No
  timestamps or random output. Matplotlib output uses fixed metadata.
- `make validate`: the CPU validation experiment, written to
  `results/kv_validation.csv`.
- `make paper`: builds `paper/main.pdf` with `tectonic`.

**`paper/main.tex`:**
- English, academic style, `article` class.
- ≥ 8 pages, ≥ 4000 words of prose.
- Sections: Abstract, Introduction, Related Work, Data & Methods, Results
  (RQ1–RQ4), Discussion, Limitations, Conclusion, and **AI-Use Disclosure**.
  The disclosure states plainly that an autonomous agent swarm (Prime Agent,
  GLM-5.3) produced this work, and what a human did.
- Figures: at least 5. Suggested:
  - KV/token vs. release date
  - KV vs. params (log-log)
  - mechanism attribution
  - capacity heatmap
  - KV-parity trend
  - validation scatter

**`paper/references.bib`:**
- ≥ 25 entries.
- Every entry is verifiable: an arXiv `eprint`, a DOI, or (≤ 35%) a live URL.
- Every entry is cited, and every citation resolves.

**Files for the human reader (in Turkish):**
- `PROGRESS.md`: a log the swarm appends to at each milestone.
- `SUMMARY_TR.md`: a final 1-page summary.

## Completion gate
```
python3 -I /home/ozkangu/Desktop/research/_gates/kv_census_gate.py /home/ozkangu/Desktop/research/kv-census
```
Run it yourself before declaring completion. Fix the work, never the gate.
