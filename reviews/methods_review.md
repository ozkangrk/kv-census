# Adversarial methods review — kv-census (statistics & inference)

Reviewer: fresh subagent (`methods-reviewer`), Phase 4. I did not write any part of
this work. I read AGENTS.md, BRIEF.md, analysis/PLAN.md, paper/main.tex,
paper/sections/*.tex, paper/generated/*, results/*, and the analysis/census code,
and re-computed the headline statistics from `data/models.csv`, the snapshotted
configs, and `results/*.csv` with the project venv (`.venv/bin/python`,
`CUDA_VISIBLE_DEVICES=""`).

Everything below was re-derived, not eyeballed. Section "Verified by recomputation"
lists what I ran and confirmed.

---

## MAJOR

### M1. Every mechanism-regression coefficient in Table 2 is mislabeled (off-by-one code bug); the paper's sign narrative is built on the mislabeled numbers

**Where.** `analysis/run_analysis.py:139`:

```python
bz, r2_full = ols(Z, y)          # line 121: bz = [intercept, b_params, ..., b_shared]  (9 values)
...
betas=dict(zip(F.columns, bz)),  # line 139: zips 8 column names with 9 values
```

`zip(F.columns, bz)` pairs each predictor name with the *previous* coefficient and
drops the last one. The value published as "total params (log10) β = 4.928"
(`paper/generated/table_mechanism.tex`; macros `\kvcBetaParams`) is the regression
**intercept** (mean of log10 KV bytes/token = 4.928); the true params coefficient
(0.408) is published as the GQA coefficient; the true GQA coefficient (−0.419) is
published as head dim; ...; the true cross-layer-sharing coefficient (−0.009) is
silently dropped.

I reproduced this exactly with `.venv/bin/python` on the same sample (N=1930):

| predictor | published β | true β (same code, `zip(..., bz[1:])`) |
|---|---|---|
| log10 params | **4.928 (intercept!)** | 0.408 |
| log(1+gqa) | 0.408 | **−0.419** |
| head_dim | −0.419 | **+0.166** |
| log(1+depth) | 0.166 | **+0.294** |
| window share | 0.294 | **−0.082** |
| recurrent share | −0.082 | **−0.214** |
| MLA latent | −0.214 | **−0.237** |
| shared KV | −0.237 | **−0.009** |

**Why it matters.** `paper/sections/results.tex:42` builds an interpretation on the
wrong numbers: "the MLA-latent indicator (β=−0.214) and cross-layer KV sharing
(β=−0.237) both lower predicted KV cost"; and especially "Two signs run against
naive per-layer accounting—GQA ratio enters positively (β=0.408) and head dimension
negatively (β=−0.419), while window share (β=0.294) and depth (β=0.166) enter
positively—because the coefficients are partial correlations conditioned on
parameter count and depth, whose VIFs (7.3 and 5.6 ...) mark substantial
collinearity." With the correct mapping **all signs agree with naive per-layer
accounting** (GQA ratio −, head dim +, depth +, window share −, recurrent share −,
latent −), so the "counterintuitive signs explained by collinearity" paragraph
explains an artifact of the bug. The VIF column, by contrast, is correctly keyed
(no bug there), so the table mixes one correct column with one shifted column.

Secondary defect in the same table: the betas are called "standardized" but `y` is
never standardized (`ols(Z, y)` with raw log10 KV/token, std(y)=0.679), so all
values carry a 0.679 scale factor. Truly standardized betas (y also z-scored) are:
params 0.601, gqa −0.617, head_dim 0.245, depth 0.433, window −0.120, recurrent
−0.315, latent −0.349, shared −0.014.

**Fix.** In `run_analysis.py` use `dict(zip(F.columns, bz[1:]))`; standardize `y`
as well if the column is to be labeled standardized; regenerate
`table_mechanism.tex`/macros; rewrite the results.tex:42 paragraph and the
corresponding Discussion sentence. R² values (0.398 / 0.792 / 0.841) are
unaffected — I verified them — so H2 stands.

### M2. The deduplication does not deduplicate geometry: RQ1/RQ2/H4 statistics average ~4.3 copies of the same architecture, and the paper's "2025 rebound" is mostly an artifact of that

**Where.** `scripts/build_models_csv.py:332`: the dedup key is
`sha1(org | name-stem | geometry_signature)` where `name-stem` only strips
trailing tokens {instruct, chat, base, it, thinking, reasoning, latest, sft,
preview} and 3–6-digit strings. Identical geometry survives dedup whenever the
names differ by anything else (seed variants, per-task fine-tunes, version
suffixes like `...-Instruct-v0.1`, cross-org twins).

`paper/sections/results.tex:13` calls the 2068 dedup rows "geometry groups";
`data_methods.tex:66` says "This leaves 2068 rows in 2068 groups" under the
heading of a set that is supposed to make "base/instruct twins with identical
geometry count once" (BRIEF.md).

**What I measured** (`.venv/bin/python`, from `data/models.csv`):
- The 2068 "groups" contain only **485 distinct (org, geometry)** combinations and
  **295 distinct geometries**. 270 within-org geometry groups hold >1 dedup row
  (1,853 of 2,068 rows); 183 geometries are shared across orgs.
- Worst case: **148 separate dedup rows carry the identical 2023-era
  Llama-2-7B / pythia-6.9b MHA geometry from EleutherAI** (112 of them dated
  2025: `deep-ignorance-*`, `pythia-6.9b-*-first-ft`,
  `PinkElephants-OpenHermes-7B-DPO`, `Llama-2-7b-hf-squaring-first-ft`, ...);
  56 rows of Llama-2-7B geometry from allenai open-instruct fine-tunes; the
  capacity table itself prints three identical Mixtral-8x22B rows
  (`mistralai/...`, `internlm/...`, `openbmb/Eurux-8x22b-kto`).
- This also means the BRIEF's exclusion rule "Exclude fine-tunes, merges, and
  quantized re-uploads from other accounts" is not applied to fine-tunes hosted
  on whitelisted publisher orgs (`data_methods.tex:7` states the rule as
  "third-party accounts", which does not match what was included).

**Statistical consequences (recomputed):**

| statistic | as published (N=2068) | one row per (org,geometry) (N=485) | one row per geometry (N=295) |
|---|---|---|---|
| median kvpp 2023 → 2024 → 2025 → 2026 (B/1e9 params) | 72087.7 / 25720.2 / **76468.6** / 7374.3 | 62932 / 16815 / 8001 / 3482 | 61171 / 14800 / 8072 / 862 |
| H1 ratio (pooled 25–26 vs 2023) | 0.413 | 0.120 | 0.072 |
| Spearman ρ (kv/token vs date), p | −0.213, 1.3e−22 | −0.347, 3.3e−15 | −0.291, 3.7e−07 |
| H4 ratio (pooled median T*) | 2.25 | 7.87 | 8.83 |
| params-only R² (H2) | 0.398 | 0.239 | 0.256 |
| MoE/dense median ratio (H3) | 0.018 | 0.037 | 0.024 |

1. **The 2025 rebound — the single most-discussed feature of RQ1
   (results.tex:15, Table 1, and the entire Discussion paragraph "The 2025 bump
   is release mix") — disappears** under geometry-level dedup: the 2025 kvpp
   median is 8,001–8,072, far *below* 2023, and the series becomes monotone. The
   published rebound is dominated by duplicate rows of one org's 2023-era
   geometries uploaded in 2025. The "many small checkpoints" explanation is
   therefore mostly an artifact of the dedup unit, not a property of the release
   stream. (The rebound also has a second, undisclosed driver: multimodal rows
   whose parameter counts include vision towers — see m4.)
2. **Pseudo-replication:** Spearman/Mann-Whitney p-values (1.3e−22, 1.1e−41,
   2.7e−99) treat ~4.3 copies of the same geometry as independent observations.
   Signs survive geometry-dedup (all still p<1e−6), but the published p-values
   are not valid as stated.
3. **H4's quantitative claim is unit-sensitive:** "grew only 2.2×" (abstract,
   results.tex:67, discussion) becomes 7.9–8.8× under geometry dedup. The
   verdict (refuted, <10×) survives every estimator I tried — published-style
   2.246, true-parity-only 2.250, markers censored at 1e7 tokens 2.398, 2026-only
   vs 2023 8.24, geometry-dedup 7.9–8.8 — but "only 2.2×" is not a robust number
   and the margin to 10× is much smaller than presented.
4. H1/H2/H3 verdicts are robust (supported / weak / supported under every dedup
   unit), which is to the authors' credit — but the *magnitudes* the paper
   quotes (0.413, the 2025 kvpp median 76468.6, the rebound narrative) are not.

**Fix.** Rebuild `dedup_group` on (org, geometry-signature) or geometry alone (the
BRIEF's stated intent — "identical geometry count once"), keep the earliest
release as representative, and rerun RQ1/RQ2 and the parity trend. Report both
units if the checkpoint-level view is wanted. Rewrite results.tex:13–17 and the
Discussion "2025 bump" paragraph around the corrected numbers, state the actual
fine-tune inclusion rule, and add the duplicate-geometry counts to Limitations.

---

## MINOR

### m1. Table 1's "rows" column is wrong (shows dedup counts, not census counts)
`paper/generated/table_overview.tex:5-10`: the caption says "Rows are checkpoints;
the deduplicated set keeps one row per geometry group", but the generator writes
the same dedup count into both columns (243/563/949/313). The census rows are
263/670/1044/354 (`\kvcYearRows*` macros; `data_methods.tex:6-8`). The table
contradicts the text it sits under. Fix the generator to emit the true per-year
census counts.

### m2. Pre-registered bootstrap CI for the slope is missing
PLAN.md (RQ2): "Report slope and R^2 with 95% CI via 1000 bootstrap resamples
(seed 0)." `run_analysis.py` computes and reports the CI for R² only
(`rq2_ci`); results.tex:38 reports the slope (0.390) with no CI. This is an
undocumented deviation from the pre-registered plan. My recompute (seed 0,
same resampling): slope CI [0.363, 0.416]. Fix: add the slope CI macro and cite
it, or record the deviation in PLAN.md's dated notes.

### m3. Bounded-cache "T*" markers pollute the parity statistics and the text misdescribes them
`analysis/kv.py` (`kv_par_context` docstring, ~line 204): for a cache that
saturates below parity, the function returns the largest window W as a *marker*
(starcoder2-3b → 4096; gemma-4 family → 512–1024); only pure-recurrent models
return `inf`. Consequences:
- `results.tex:67` "A small and growing fraction of releases have no finite
  parity point at all — 0.5% in 2023 versus 3.9% in 2026 — because their caches
  saturate (fully windowed **or recurrent**)" is wrong: fully-windowed saturating
  models *do* get a finite recorded T* (the window size), so the quoted inf share
  counts pure-recurrent models only (28 rows), while 13 windowed models
  (starcoder2 ×3, gemma-4 ×5, ...) enter the yearly medians and Figure 5 with
  fake finite T* = W.
- The "KV-dominant at 32k" share (T*<32768) classifies these models as
  KV-dominant although their bounded cache can never exceed the weights
  (starcoder2-3b: cache caps at ≈0.35 GiB vs ≈6.2 GiB weights).
I recomputed: excluding the markers changes the pooled ratio only in the 4th
digit (2.2498 vs 2.2457) and the dom shares by ≤2.6pp, so no verdict moves — but
the statistic as defined includes values that are not parity contexts, and the
sentence explaining the inf share is factually wrong about its own data. Fix:
flag never-parity rows (windowed or recurrent) as censored, exclude them from
medians/IQR/dom shares, and rewrite the sentence.

### m4. Multimodal rows: KV from the text backbone, parameters including the vision tower
`data_methods.tex:8` says multimodal models "enter through their text backbone:
we parse the language-model sub-configuration and ignore vision and audio
towers", but `total_params_b` comes from Hub safetensors metadata and *includes*
the tower (the builder flags this in `notes` as `params_include_vision_tower` —
192 census rows, 169 dedup rows, all of them 2024–2026). `kv_per_param`, the
params regression, H1's pooled medians, and MoE/dense medians therefore divide
language-backbone KV by tower-inflated parameter counts for ~8% of the dedup
set, concentrated in the years the trend claims are about, biasing per-parameter
KV downward. The flag never surfaces in the paper. Fix: use language-backbone
params for per-parameter statistics (or exclude MM rows), and disclose whichever
choice is made; for capacity, total weights are the right choice — say so
explicitly.

### m5. Silent `fillna(0)` corrupts the GQA feature for 19.5% of the regression sample
`run_analysis.py:116` `F = pd.DataFrame(feats).fillna(0.0)`: 376 of 1930 rows
(mostly `mha+legacycfg`: Pythia/GPT-2-era configs where `num_attention_heads`
did not reach the CSV) get GQA ratio = log1p(0) = 0 instead of their true value
(1 for MHA → log1p(1) = 0.693). Imputing ratio=1 for those rows raises R²_full
0.841 → 0.895 and moves betas (gqa −0.617, head_dim +0.215 properly
standardized). Not documented anywhere. Fix: impute ratio=1 (or drop), and
document.

### m6. The cross-layer-sharing indicator is estimated from 2 observations
`shared=True` occurs in 2 of 1930 regression rows. Even after fixing M1 (true
coefficient ≈ −0.009, i.e. nothing), any sentence attributing an effect to
cross-layer KV sharing (results.tex:42 "cross-layer KV sharing (β=−0.237) both
lower predicted KV cost"; Discussion "sharing one cache across layers") is
unsupported by this regression. Fix: drop the indicator from the table or mark
it n=2 and rest the claim on the validation/geometry arithmetic instead.

### m7. "74 census models exceed 128 GiB on weights alone even at 4-bit" — wrong description of the number
`results.tex:58`. The macro counts dedup groups with known params whose 4-bit
weights **plus the 6 GiB overhead** exceed 128 GiB (74); the weights-alone count
is 64 (I recomputed both from params). `discussion.tex` states it correctly
("under the modeled 6 GiB runtime overhead"); results.tex contradicts it. Also
"census models" → these are dedup groups with known params (N=1971), not census
rows (2331). Fix the sentence.

### m8. Regression N does not add up from the stated exclusion
`results.tex:40`: "The regression uses N=1930 groups after excluding 67
zero-marginal-KV rows". 2068 − 67 = 2001 ≠ 1930; another 71 rows are dropped for
missing parameter counts (never mentioned here). State both exclusions.

### m9. Rank-biserial sign convention unstated
`results.tex:44` reports "rank-biserial −0.850" without saying for which group.
The code computes 1 − 2U/(n₁n₂) with U from the dense sample, i.e. the rank
biserial *for the MoE group* (negative = MoE lower). One clause fixes it.

### m10. Concurrency is promised, pre-registered, computed — and never reported
The abstract asks "How much context and concurrency fit fixed memory budgets?";
PLAN.md pre-registers "max concurrent sequences at 32k and 128k"; the capacity
CSV contains `conc_32k`/`conc_128k` for every cell; but no concurrency number
appears anywhere in the paper (grep: only abstract/introduction mention
concurrency). Either report it (a small table or a sentence with macro-backed
numbers) or soften the abstract's claim.

### m11. Capacity solver returns a positive token count when not even the recurrent state fits
`run_analysis.py` `_solve_bp`: when 0 < pool ≤ S (fixed state), the crossing
search never triggers and the function returns the last window breakpoint
instead of 0. Real occurrence: `Qwen/Qwen3.5-9B-Base` at 24 GiB/BF16 has
pool ≈ 21.1 MB < S ≈ 26.3 MB, and the CSV reports max_tokens = 486.58 (BF16 KV)
vs 0.0 (FP8 KV) for the same budget — mutually inconsistent and both wrong
(answer: nothing fits). 2 of 35,478 rows affected; conclusions unaffected. Fix:
return 0 when bytes(T=0) = S ≥ pool, and add a unit test.

### m12. The documented "objective selection rule" does not match the implemented one
`run_analysis.py` module docstring (lines 1–9, referenced by the PLAN deviation
note) says the capacity table takes rows "sorted by kv_bytes_per_token_bf16
descending, first N" plus a 2023 anchor; `main()` actually collapses by
(org, layers, kv heads, head dim, attention type, window), then takes top-12 by
params + top-12 by kv/token among feasible rows ≥2024, plus the 2023 min-kv/param
anchor (which lands on a zero-KV fully-windowed model, starcoder2-3b — legal but
worth knowing). The rule is objective, but the docstring the plan points to is
inaccurate, and the collapse still admits cross-org duplicates (three identical
Mixtral-8x22B rows in Table 3). Update the docstring and dedup the table rows by
geometry.

---

## NOTES (opinion)

- **n1.** Geometry-only R² = 0.792 is close to true by construction: the outcome
  is an exact function of the geometry predictors (2·L·n_kv·d_h, windows,
  latents), so "geometry explains most of the variance" is only news relative to
  parameter count. The params-vs-geometry contrast is the right one; consider
  saying this explicitly.
- **n2.** `make paper` is not byte-reproducible (a rebuild changed main.pdf
  263,108 → 263,105 bytes; the committed PDF was already dirty vs HEAD when I
  started). Only `make analysis` determinism is required, and it holds — my
  re-run left results/, paper/figures/, paper/generated/ byte-identical.
- **n3.** Validation compares float32 analytic vs float32 measured while the
  census is BF16. Internally consistent and disclosed; real serving at BF16
  halves the bytes. Fine as a check of the accounting, worth one sentence.
- **n4.** `attention_type` labels include a `+legacycfg` suffix class (363 rows
  in the regression sample) that the paper's labeling paragraph does not
  mention; harmless but the "labels are honest by construction" claim should
  cover it.
- **n5.** Abstract: "Analytic estimates match 6 of 7 validation rows across 6
  attention classes within 1%" can be misread as "6 classes within 1%". Also
  "computing KV bytes per token from every public model config" — 9 configs
  failed to parse (disclosed in Methods); soften "every".
- **n6.** Fig 4 renders infeasible cells (NaN → 0 → log10(1)) identically to a
  zero/one-token cell; cosmetic.

---

## Verified by recomputation (what I ran)

All with `.venv/bin/python` (or kernel pandas/scipy on the same CSVs),
`CUDA_VISIBLE_DEVICES=""`, CPU only. Scratch scripts removed afterwards.

1. **RQ1 (exact match to macros):** per-year dedup counts (243/563/949/313),
   median KV B/token (327680/131072/70272/36864), median kvpp×1e9
   (72087.7/25720.2/76468.6/7374.3), Spearman ρ/p for both outcomes
   (−0.213, 1.3e−22; −0.298, 1.1e−41), H1 ratio 0.4133 → supported.
2. **RQ2:** OLS slope 0.3899, R² 0.3976, bootstrap R² CI [0.350, 0.448]
   (seed 0, 1000 resamples, reproduced exactly); slope CI [0.363, 0.416];
   mechanism R² 0.7915/0.3976/0.8408; VIFs (7.34/2.00/2.26/5.57/1.07/1.10/1.20/1.00);
   H3: medians 62940.6 vs 1110.5, ratio 0.0176, Mann–Whitney p = 2.7e−99,
   rank-biserial −0.850 (N=1695/235) — all match the published macros.
3. **H4 robustness:** recomputed T* per row from the snapshotted configs via
   `analysis/kv.py` (per-year medians match: 26729/71040/26155/220156.5); verdict
   "refuted" holds under every estimator tried (published-style 2.25, true-parity
   only 2.25, censored-at-1e7 2.40, 2026-only 8.24, geometry-dedup 7.9–8.8) —
   but see M2/M3 on how fragile the "2.2×" magnitude is.
4. **Capacity quotes:** OLMo-7B 24/BF16 = 10,588 ("11k"), 24/FP8 = 23,726
   ("24k"), 128/BF16 = 223,580 ("224k"), 128/FP8 = 236,718 ("237k");
   gpt-oss-120b 128/4bit = 1,770,708 ("1.8M"); Mistral-Large 23k/172k;
   infeasible-at-128/4bit = 74 (with 6 GiB overhead) vs 64 weights-alone — all
   from `results/rq3_capacity_overhead6.csv`.
5. **models.csv integrity:** every `mha/gqa/mqa` row satisfies
   2·L·n_kv·d_h·2 exactly (1519/1519); kv_bytes_at_32k/128k recomputed from
   configs via kv.py match the CSV exactly for gpt-oss-20b, starcoder2-3b,
   granite-4.0-h-micro.
6. **Counts:** 2331 census rows, 2068 dedup, 33 orgs, 237 gated (google 130,
   meta-llama 42 — consistent with "no Llama rows"), 104 no-params, 9 kv-config
   errors; 107 test functions and 61 distinct fixture configs used in
   tests/test_kv.py (paper's claims); transformers 5.19.0 in requirements.txt
   matches the "transformers 5.19" statement.
7. **Determinism:** `make analysis` re-run left results/, paper/figures/,
   paper/generated/ byte-identical to HEAD (only the pre-existing dirty
   main.pdf and my reviews/ scratch showed in `git status`).
8. **Pre-registration:** `git log --follow analysis/PLAN.md` shows PLAN.md
   committed in Phase 0 and only appended-to (deviation notes) in Phase 2;
   thresholds were not edited after the fact.

---

## Verdict

**MAJOR REVISIONS.**

The parts that matter most are solid: the pre-registration is real and was not
retro-edited, the analytic accounting is exact where checkable, `make analysis`
is deterministic, the capacity arithmetic checks out to the token, and all four
hypothesis verdicts (H1 supported, H2 weak, H3 supported, H4 refuted) survive
every robustness probe I threw at them, including a full re-deduplication of the
census. The honesty framing (refuted H4 reported as a finding) is genuine.

But two defects are disqualifying as the paper stands: (1) the mechanism
regression table is numerically mislabeled by an off-by-one `zip`, and the
results paragraph explaining the "counterintuitive signs" interprets that bug;
(2) the deduplication unit leaves ~4.3 copies of each geometry in the trend
statistics, which manufactures the paper's most-discussed feature (the 2025
per-parameter rebound), invalidates the published p-values as independence
claims, and makes the H4 headline "only 2.2×" a unit-dependent number (7.9–8.8×
under geometry dedup, still refuted). Both are fixable without new data collection:
fix the zip, rebuild dedup on geometry, rerun `make analysis`, and rewrite the
affected paragraphs (results.tex:13–17, 38–44, 67; discussion.tex:20; the Table 1
generator). The minor items (Table 1 "rows" column, missing slope CI, T* markers,
multimodal params, fillna, n=2 sharing indicator, "74 on weights alone",
concurrency reporting, solver boundary case, docstring mismatch) should be fixed
in the same pass.
