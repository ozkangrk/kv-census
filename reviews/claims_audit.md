# Claims audit — kv-census paper

Reviewer: `claims-auditor` (fresh adversarial subagent; did not author any part of the paper).
Scope: sentence-level audit of every number, quantitative adjective, and citation in
`paper/main.tex` and `paper/sections/*.tex`, checked against `paper/generated/*.tex`,
`results/*`, `analysis/PLAN.md` (pre-registered plan), `paper/references.bib`,
`notes/related_work.md`, and the cached retrieval records in `data/raw/literature/`.
Also re-verified reproducibility claims directly (see "What was verified sound").

Severity: **MAJOR** = wrong/misleading claim, statistical error, unsupported conclusion,
gate-visible defect. **MINOR** = unclear, imprecise, or improvable. **NOTE** = opinion.

## What was verified sound (so the authors know what was checked, not just what failed)

- Every macro in `paper/generated/macros.tex` reproduces from `results/rq1_stats.csv`,
  `results/analysis_stats.json`, `results/dedup_rows.csv`, `results/kv_validation.csv`,
  `data/gated_skipped.txt`, `data/kv_config_errors.txt`. I independently recomputed the RQ1
  medians, the H1 ratio (0.4133), and both Spearman correlations (−0.213 / −0.298) from
  `results/dedup_rows.csv`; they match the paper exactly.
- `make analysis` re-runs byte-identical (17 output files hashed before/after; zero changes);
  `make test` → 107 passed. "107 unit tests" and "61 real configuration snapshots"
  (`tests/test_kv.py`, `tests/fixtures/raw_configs/`) and `transformers==5.19.0`
  (`requirements.txt`) are accurate.
- Capacity-table cells spot-checked against `results/rq3_capacity_overhead6.csv`: OLMo-7B
  11k/24k/224k/237k, gpt-oss-120b 1.8M, Ling-3.0-flash 431k/7.3M, starcoder2-3b ∞, and the
  infeasible count 74 at 128 GiB/4-bit/6 GiB all match.
- RQ4 claims match `results/kv_validation.csv`, including the *sign* of the windowed gaps
  (measured < analytic ⇒ "small negative deviations" is correct) and the conv-state padding
  arithmetic on the mamba row (+20480 B = 5120 × 1 × 4 B).
- Bibliography: 34 entries, 34 cited, 0 unresolved, all arXiv eprints; titles match the cached
  arXiv Atom records in `data/raw/literature/`. The claim-support mapping in
  `notes/related_work.md` holds for the overwhelming majority of citation sentences.
- Counted limitations are real: 237 gated skips, 9 parse errors, 104 no-param rows; no
  `meta-llama` org rows; 5 `google` rows are all gemma-4; 2025 cohort is demonstrably
  small-model-heavy (median 3.6B vs 6.9B in 2023; 51.5% < 4B vs 30.9%).
- Paper builds (15 pages), 6 figures exist, ~5.2k words, AI-use disclosure matches the
  process recorded in `PROGRESS.md` (swarm-produced, human passive).

## Findings

| # | Location | Sentence excerpt | Problem | Severity | Fix |
|---|----------|------------------|---------|----------|-----|
| 1 | `paper/generated/table_overview.tex:5-10` (Table 1) + `analysis/run_analysis.py` `make_tex` | `year & rows & dedup rows` … `2023 & 243 & 243` | The "rows" column is the **dedup** per-year count (243/563/949/313, sum 2068), printed twice; it contradicts `data_methods.tex:10` ("263 released in 2023, 670 in 2024, 1044 in 2025, 354 in 2026" — sum 2331 = `\kvcCensusRows`) while the caption says "Rows are checkpoints". Generator bug: `make_tex` writes `r1[y]['n']` into both columns. Readers cannot reconcile 263 vs 243. | **MAJOR** | In `make_tex`, feed raw per-year census counts (the `kvcYearRows*` source) into the "rows" column and keep `r1[y]['n']` for "dedup rows"; or drop the duplicate column. Regenerate via `make analysis`. |
| 2 | `data_methods.tex:64` "checkpoints with identical attention geometry share a `dedup_group`"; `limitations.tex:31` "collapses checkpoints with identical geometry into one row per group"; `results.tex:13` & `conclusion.tex:4` "2068 geometry groups" | "…identical attention geometry share a dedup_group…" | The implemented rule (`scripts/build_models_csv.py`) is `sha1(org \| name-stem \| geometry_signature)`: identical geometry is **necessary but not sufficient**. Measured: the 2068 "geometry groups" realize only **299 distinct geometries**; one geometry (32L/32H/128hd MHA, 524288 B/token) is shared by **260 groups**, including ~112 EleutherAI 2025 Pythia-geometry research variants that each count separately in RQ1/RQ2. Geometry-only grouping, as described, would materially change the 2025 medians the paper explains as "release mix". `analysis/PLAN.md` uses the same wording, and no dated deviation note records the org+stem restriction. | **MAJOR** | Describe the implemented rule precisely (org + stripped family stem + geometry; suffix tokens and date-ish tails collapsed) and add a dated deviation note to `PLAN.md`; rename "geometry groups" to "dedup groups" in `results.tex:13`, `conclusion.tex:4`, and elsewhere. Optionally report the 299-distinct-geometry figure so readers see the true diversity. |
| 3 | `data_methods.tex:85` | "Runtime overhead is a free parameter: we report the full grid over 0, 2, 6, and 12 GiB…" | False as a statement about the paper: no overhead other than 6 GiB appears in any table or figure. It is also false about the repo: `make analysis` exports only two slices (`rq3_capacity_overhead6.csv`; `rq3_sensitivity_overhead.csv` = 128 GiB/BF16/BF16 only). `PLAN.md` pre-registered "sensitivity analysis in the paper" for overhead. `PROGRESS.md` (Phase 2) claims the full grid ships "under results/"; it does not. | **MAJOR** | Either add the pre-registered overhead-sensitivity output (export the full grid to `results/` and one small table or a sentence with the range of max-tokens across 0–12 GiB for the Table-3 models — all within the registered plan), or reword to "we compute the full grid … and report 6 GiB in the headline tables" and fix the `PROGRESS.md` claim. |
| 4 | `results.tex:38` | "The regression uses $N=1930$ groups after excluding 67 zero-marginal-KV rows…" | Arithmetic does not close: 1930 + 67 = 1997 ≠ 2068 (nor 1971). Actual exclusions: 97 no-param rows + 41 zero-KV rows *with* params (67 zero-KV rows exist, but 26 of them lack params and were already out). | MINOR | "…after excluding 97 rows without parameter counts and 41 zero-marginal-KV rows (67 zero-KV rows exist in total)". |
| 5 | `results.tex:58` | "…and 74 census models exceed 128 GiB on weights alone even at 4-bit" | Strictly wrong for 10 of the 74: their 4-bit weights are ≤ 128 GiB (≥122 GiB) and they fail only after the 6 GiB overhead. `discussion.tex:42` states it correctly ("under the modeled 6 GiB runtime overhead"). Also "census models" — the 74 are dedup rows with known params, not all 2331 census rows. | MINOR | Align with the discussion wording: "74 deduplicated models leave no KV pool at 128 GiB even at 4-bit weights once the 6 GiB overhead is charged". |
| 6 | `abstract.tex:19`; `conclusion.tex:17-18` | "Analytic estimates match 6 of 7 validation rows across 6 attention classes within 1%" | The 6 within-1% rows span 5 classes (mha, gqa, gqa+swa, mla, hybrid-mamba); the 6th class (pure recurrent) is exactly the >1% row. The compressed phrasing mis-assigns class coverage to the matched subset. | MINOR | "…match 6 of 7 validation rows within 1%; the validation set spans 6 attention classes" (as `introduction.tex` already phrases it). |
| 7 | `introduction.tex:10` | "DistServe replicates the KV cache when it splits prefill from decoding" | Not supported by the cached record (`arxiv_2401.09670.xml`): the abstract describes disaggregating prefill/decoding onto different GPUs and *minimizing* the communication caused by disaggregation — nothing about replicating the KV cache. | MINOR | "DistServe splits prefill and decoding onto different GPUs, making KV-cache placement a first-class concern". |
| 8 | `related_work.tex:39` | "Layer-condensed KV caches only a small number of **bottom** layers" | The cached abstract (`arxiv_2405.10637.xml`) says "a small number of layers" only; "bottom" is an unsupported detail (and is not how CLA is usually described). | MINOR | Drop "bottom": "caches KV for only a small number of layers and reads the rest from that cache". |
| 9 | `introduction.tex:29` | "measured runtime comparisons on such hardware already **rank models** by KV and prompt-cache behavior" | The cached record (`arxiv_2511.05502.xml`) ranks **runtimes** (MLX, MLC-LLM, llama.cpp, Ollama, MPS) on a single model family (Qwen-2.5), not models. The related-work phrasing is accurate; this one is not. | MINOR | "…already measure KV and prompt-cache behavior as first-class metrics" (mirroring `related_work.tex`). |
| 10 | `limitations.tex:7` | "…so the census contains **no Llama rows at all**" | True only for the `meta-llama` org. The census contains Llama-*derived* checkpoints, two of them in Table 3 (`deepseek-ai/DeepSeek-R1-Distill-Llama-70B`, `nvidia/OpenMath-CodeLlama-13b/7b-Python-hf`, `EleutherAI/PinkElephants-OpenHermes-13B-DPO`). A reader would conclude Llama geometry is absent from the data. | MINOR | "…the census contains no rows published by meta-llama; only third-party Llama-derived fine-tunes (e.g. distills) appear". |
| 11 | `results.tex:42` | "the coefficients are partial correlations conditioned on parameter count and depth" | They are standardized OLS coefficients (which can exceed 1 under collinearity, hence β=4.928), not partial correlations. Also "GQA ratio ($\log(1+r)$)" never defines the direction of $r$; the code uses $r = q\text{-heads}/\text{kv-heads}$ (`analysis/kv.py` `describe`), so "enters positively" is unreadable without the source. | MINOR | "standardized coefficients"; add "$r=$ query heads per KV head" to the table/methods. |
| 12 | `results.tex:52` (Fig. 4 caption) + `analysis/run_analysis.py` module docstring | "rows follow the objective selection rule documented in the analysis code" | The documented rule ≠ the implemented rule. Docstring: "sorted by kv_bytes_per_token_bf16 descending, first N; plus the 2023 … LOWEST kv-per-param anchor". Code (`main`): collapse by `(org, layers, kvh, hd, attention_type, window)` keeping earliest, then top-12 by params + top-12 by kv/token (year ≥ 2024, feasible at 128/4-bit) + the anchor. The anchor (`starcoder2-3b`, the ∞ row the text spotlights) is a narrative-chosen showcase. | MINOR | Update the docstring to the implemented rule (or vice versa); consider saying "a documented, code-fixed selection rule" without "objective", and disclose the anchor choice in the caption. |
| 13 | `ai_disclosure.tex:11` | "every quantitative statement in this paper expands from macros generated by a deterministic, offline analysis pipeline" | Overclaims: "107 unit tests", "61 real configuration snapshots" (`data_methods.tex:58`), "transformers 5.19" (`results.tex:81`) are typed prose constants (they trace to `tests/`/`requirements.txt`, not `paper/generated/`), and the RQ3 prose numbers (11k/24k/224k/237k/1.8M) are typed but do match generated table cells. | MINOR | "every **result** number expands from generated macros or table cells; tool-version and test counts are verifiable from the repository". |
| 14 | `data_methods.tex:82-83` | "…including 16 group-scale bits per 32-element group \cite{frantar2022gptq}"; "priced at 2 or 1 byte per element … \cite{liu2024kivi}" | The retrieved GPTQ abstract does not mention group-scale overhead; the 4.5-bit convention is a plan constant, not a GPTQ result. The KIVI citation is attached to plain BF16/FP8 byte sizes the abstract does not discuss. Cite only what the record supports. | MINOR | Attach the 4.5-bit convention to the plan ("as pre-registered") and keep \cite{frantar2022gptq} only for the existence of 4-bit weight quantization; drop or move \cite{liu2024kivi} to the KV-quantization sentence where it belongs. |
| 15 | Pre-registered reporting gaps (`analysis/PLAN.md` vs paper) | abstract.tex:10 & introduction RQ3 bullet advertise "context **and concurrency**"; PLAN RQ3 requires "max concurrent sequences at 32k and 128k"; PLAN RQ1 requires "median **and IQR** by release year"; PLAN RQ2 requires "incremental $R^2$ of each mechanism group" | All computed (`conc_32k`/`conc_128k` in `results/rq3_capacity_overhead6.csv`; `iqr_*` in `rq1_stats.csv`) but none reported anywhere in the paper; Fig. 3 shows params-only / geometry-only / full $R^2$ rather than the registered incremental bars. The abstract asks a question (concurrency) the results never answer with a number. | MINOR | Add a sentence in RQ3 with e.g. median concurrency at 32k for a couple of named models, add IQR columns to Table 1 (or a parenthetical), and either add the registered incremental-$R^2$ numbers or record a dated note in `PLAN.md` that the three-set comparison supersedes them. |
| 16 | `abstract.tex:8` | "computing KV bytes per token from **every public model config**" | Overbroad: 237 gated repos skipped, 9 configs unparseable, third-party accounts excluded by design. | NOTE | "from the public config of every included checkpoint". |
| 17 | `results.tex` RQ2 / `discussion.tex` ¶1 | "$R^2=0.792$ for geometry and mechanisms alone vs $0.398$ for parameters alone" | The comparison is 7 predictors vs 1, and the geometry features enter the outcome near-mechanically (KV/token is a function of layers × KV heads × head dim). Pre-registered as such, but one caveat sentence would preempt the objection. | NOTE | Add: "geometry uses seven predictors where parameters use one, and the outcome is partly a deterministic function of the geometry features; the comparison is pre-registered, not symmetric." |
| 18 | `results.tex:67` | "A small and **growing** fraction … no finite parity point at all---0.5% in 2023 versus 3.9% in 2026" | Non-monotone (1.5% in 2024, 0.7% in 2025); the two-endpoint phrasing hides the dip the paper itself stresses elsewhere. | NOTE | "…fluctuating but rising overall (0.5% → 1.5% → 0.7% → 3.9%)". |
| 19 | `data_methods.tex:66` | "This leaves 2068 rows in 2068 groups" | Trivially identical by construction (one row per group); reads as if two quantities were being distinguished. | NOTE | "This leaves 2068 deduplicated rows (one per group)". |
| 20 | `ai_disclosure.tex:14` | "metadata retrieved from the arXiv, OpenAlex, **and Crossref** APIs" | The cache evidences arXiv (48 Atom XMLs) and one OpenAlex search; no Crossref responses are cached. All 34 entries are arXiv-verified, so the tri-API claim overstates. | NOTE | "retrieved from the arXiv API (with OpenAlex used for discovery) and cached in the repository". |
| 21 | `limitations.tex` (Hub-frame paragraph) | "the Hub is a biased frame … recent release years skew toward many small checkpoints" | Org concentration is not disclosed: allenai (610) + EleutherAI (383) = 43% of census rows; in 2025 they are 54% of the cohort, and 112 of the 148 EleutherAI 2025 dedup rows are Pythia-6.9B-geometry variants. This materially shapes the "release mix" story and deserves one sentence. | NOTE | Add to Limitations: "two publishers (allenai, EleutherAI) contribute 43% of census rows and over half of the 2025 cohort, many of them research variants sharing one geometry". |

## AI-Use Disclosure accuracy

Substantively accurate: the work products (census code, data collection, analysis, drafting)
were demonstrably swarm-produced per `PROGRESS.md`; the gate really does re-run `make analysis`
and require byte-identical outputs (verified in the gate source and by my own re-run); the
analysis plan really predates the trend results with dated deviation notes; refuted H4 is
reported as refuted. Two overclaims, listed above: "every quantitative statement expands from
macros" (#13) and the tri-API retrieval phrasing (#20). The human's role (brief, compute,
sandbox, passive supervision) matches everything observable in the repository.

## Verdict

**MAJOR REVISIONS.** The quantitative pipeline is solid and honestly reported at the verdict
level — every macro reproduces, the analysis is byte-deterministic, refuted hypotheses stay
refuted — but three claims misdescribe what was done or shown: the census-overview table
contradicts the prose counts (#1), the deduplication is described as geometry-identity grouping
while the code groups by org + name-stem + geometry (#2), and the paper claims to report an
overhead-sensitivity grid it never reports (#3). All three fixes are localized (one generator
line, one method paragraph + `PLAN.md` deviation note, one table/sentence or rewording); none
require new data collection. With #1–#3 and the citation-support minors (#7–#9) fixed, this
would move to MINOR REVISIONS.
