# Pre-registered analysis plan (fixed BEFORE any trend results were inspected)

Written in Phase 0 per AGENTS.md. Do not change thresholds after `data/models.csv`
statistics have been computed. Any deviation must be recorded in this file with a
dated note and a reason.

## Population and units
- Unit of analysis: one HF checkpoint row in `data/models.csv` (>= 150 rows).
- Trend statistics (RQ1) and cross-model regressions (RQ2) use the DEDUPLICATED set:
  exactly one row per `dedup_group` (the representative is the row with the earliest
  `release_date`, ties broken by lexicographic `repo_id`).
- Capacity computations (RQ3) use all rows (each model is its own capacity point).
- `release_date` = Hub `createdAt` day. Limitation, stated in the paper.

## RQ1 — trend in KV cost, 2023-2026
Primary outcomes (BF16):
- `kv_bytes_per_token_bf16` (asymptotic marginal, unbounded-cache layers only),
- `kv_per_param_bf16 = kv_bytes_per_token_bf16 / (total_params_b * 1e9)`.
Statistics, computed on the dedup set and also reported per year:
- median and IQR by release year (2023, 2024, 2025, 2026-to-date);
- Spearman rho of each outcome vs `release_date` ordinal, with two-sided p;
- H1 ("median per-parameter KV cost fell substantially") is SUPPORTED iff
  median(kv_per_param_bf16, 2025+2026 pooled) <= 0.75 * median(kv_per_param_bf16, 2023).
  Otherwise REFUTED (or PARTIAL if 0.75 < ratio <= 0.90; report the ratio either way).

## RQ2 — mechanisms vs parameter count
- H2: regress log10(kv_bytes_per_token_bf16) on log10(total_params_b) on the dedup set.
  "Weak predictor" iff R^2 < 0.5. Report slope and R^2 with 95% CI via 1000 bootstrap
  resamples (seed 0).
- Mechanism model: OLS on log10 KV/token with predictors:
  log10(gqa_ratio+1), head_dim, log10(num_layers+1), window_share = n_window_layers/num_layers,
  recurrent_share, latent indicator (n_latent_layers>0), kv-shared indicator,
  log10(total_params_b). Report standardized betas and incremental R^2 of each
  mechanism group (params-only vs geometry-only vs both).
- H3: MoE vs dense on the dedup set: median kv_per_param_bf16 per group, Mann-Whitney U
  two-sided p, rank-biserial effect size. SUPPORTED iff MoE median <= 0.5 * dense median.
- VIF table for the mechanism model (collinearity honesty).

## RQ3 — capacity under fixed budgets
Parameters (sensitivity grid, all reported):
- budgets: 24, 80, 128 GiB (2^30 bytes);
- weight precision: BF16 (2 B/param), FP8 (1 B/param), 4-bit (4.5 bits/param effective =
  4 + 16/32 group-scale overhead, stated explicitly);
- KV precision: BF16 (2 B/elem), FP8 (1 B/elem);
- runtime overhead: 0, 2, 6, 12 GiB (parameter; sensitivity analysis in the paper,
  headline tables use 6 GiB).
Per (model, budget, weight-precision, kv-precision, overhead):
- max total KV tokens = (budget - weights - overhead) / (kv_bytes_at(T=1 token)) — solved
  accounting for window saturation and recurrent state: exact numeric solution on a
  T grid (log-spaced 2^10..2^20, linear interpolation, monotone);
- max concurrent sequences at 32k and 128k;
- KV-parity context `T*` (kv_bytes_at(T*) = weight bytes at BF16 weights);
  median T* per release year => H4 SUPPORTED iff median(T*, 2025+2026) >= 10 * median(T*, 2023);
- KV-dominant regime boundary: for each model, the T above which KV bytes exceed
  weight bytes (same T*); report the fraction of models with T* < 32k per year.

## RQ4 — validation
- `results/kv_validation.csv` per INTERFACES.md; success criteria: >= 6 models, >= 4
  attention classes, >= 5 rows within 1%; every row > 1% carries an explanation
  (e.g. MLA decompressed in HF cache, window allocated in full, conv-state padding).
- Reported in the paper as a scatter (analytic vs measured, log-log) plus a table.

## Figures (all PDF, fixed metadata, no timestamps; seeds fixed)
1. KV/token vs release date (dedup set, log y, colored by attention_type).
2. KV/token vs total params, log-log, with the fitted line and R^2 (H2).
3. Mechanism attribution: incremental R^2 bars (params-only, +geometry, +mechanisms).
4. Capacity heatmap: models x (budget, weight-precision) -> max total tokens at
   128 GiB & 24 GiB, overhead 6 GiB.
5. KV-parity context trend by release year (median + IQR).
6. Validation scatter analytic vs measured.
Optional 7th: KV-dominant boundary fraction by year.

## Determinism
- `analysis/run_analysis.py` reads only `data/` and `results/`, writes only `results/`,
  `paper/figures/`, `paper/generated/`. Sorted iteration, seed 0, `float` formatting via
  f-strings with fixed precision; matplotlib PDF metadata `CreationDate: None`
  (`matplotlib.rcParams['pdf.compression'] = 6`, env `SOURCE_DATE_EPOCH=0`).
- The completion gate re-runs `make analysis` and requires byte-identical outputs.

## Honesty rules (from AGENTS.md)
- Gated-skip count and Hub coverage bias go into Limitations, with exact counts from
  `data/gated_skipped.txt` and `data/raw/hub_api/`.
- Refuted hypotheses are reported as refuted, in the abstract if primary.
