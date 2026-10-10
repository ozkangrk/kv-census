# Review responses (Phase 4)

Reviews: `reviews/methods_review.md` (2 MAJOR, 12 MINOR, 6 notes),
`reviews/claims_audit.md` (3 MAJOR, 11 MINOR, 7 notes). Every MAJOR/MINOR was fixed or
explicitly addressed below; the completion gate passes 6/6 after the fixes (16-page PDF).
All fixes trace to regenerated macros/tables (`make analysis`, deterministic) or to
section text; no number is hand-typed.

## methods-reviewer

- **M1 (beta labels off-by-one; mislabeled intercept as params beta; y not standardized):**
  FIXED. `run_analysis.py` now zips `bz[1:]` and z-scores the outcome; Table 2 and macros
  regenerated (params 0.601, gqa -0.617, ...); the "two signs run against naive accounting"
  paragraph was wrong and is rewritten: with correct labels every sign matches naive
  per-layer accounting. R^2 values were unaffected (reviewer-verified).
- **M2 (dedup does not deduplicate geometry; rebound artifact; pseudo-replicated p-values):**
  ADDRESSED as a reporting correction plus robustness analysis. The pre-registered
  family-level unit stays primary (pre-registration honored), and every affected statistic
  is now also reported under one-row-per-geometry deduplication (292 geometries, 482
  org-geometry combos): Table 1 gains geo columns; the 2025 rebound is re-attributed to
  duplicated-geometry fine-tune rows (worst cluster \kvcWorstGeoCluster{} rows,
  \kvcWorstGeoClusterRecent{} released 2025+) with the monotone geo series quoted; H1
  (0.413 family / 0.071 geo) and H4 (2.2 family / 11.0 geo) verdict sensitivity is stated
  in the abstract, results, discussion, and conclusion. Methods and Limitations describe
  the actual dedup key and its failure mode; the fine-tune inclusion rule is stated.
- **m1 (Table 1 rows column):** FIXED (true per-year census counts; dedup and geo columns
  separate; IQR column added).
- **m2 (slope CI missing):** FIXED (\kvcParamsSlopeLo/Hi from the pre-registered seed-0
  bootstrap; cited in text).
- **m3 (bounded-cache T* markers pollute parity stats; wrong inf-share text):** FIXED.
  `parity_Tstar` censors never-parity models (asymptotic cache below weight bytes) to
  infinity; medians/IQR/dominant shares exclude them; the inf-share sentence now states
  the censoring semantics correctly.
- **m4 (multimodal params include vision towers):** DISCLOSED + robustness cut. Methods
  states the inflation (\kvcMMRows{} rows, flagged in `notes`), capacity keeps total
  params (correct for memory budgets), and RQ1 reports the multimodal-excluded ratio
  (\kvcHOneRatioNoMM) — which flips H1; the abstract and conclusion say so explicitly.
- **m5 (silent fillna(0) corrupts GQA feature):** FIXED with explicit imputation (ratio=1
  where KV heads are absent/equal to attention heads), documented in Methods; betas and
  R^2 updated.
- **m6 (shared-KV indicator n=2):** FIXED. n is printed in Table 2 and the text no longer
  rests any claim on that coefficient.
- **m7 ("74 exceed weights alone" wrong):** FIXED. \kvcWeightsAloneBig{} weights-alone vs
  \kvcInfeasibleBigModels{} with the 6 GiB overhead; sentence corrected ("dedup groups
  with known parameters", not "census models").
- **m8 (regression N arithmetic):** FIXED. 97 no-parameter + 67 zero-marginal-KV groups,
  26 of them overlap: 2068 - 138 = 1930.
- **m9 (rank-biserial convention):** FIXED (direction stated: negative = MoE ranks lower).
- **m10 (concurrency computed but unreported):** FIXED. Median concurrency at 128 GiB
  (32k: \kvcConcMedThirtyTwoK; 128k: \kvcConcMedOneHundredTwentyEightK;
  \kvcConcMedOneHundredTwentyEightKZeroShare% cannot hold one 128k context) reported;
  sensitivity table added; the FULL grid (budgets x weight precision x KV precision x
  overhead) is exported as `results/rq3_capacity_full.csv`.
- **m11 (solver returns tokens when even the state does not fit):** FIXED (`_solve_bp`
  returns 0 when the fixed state alone exceeds the pool; full grid regenerated).
- **m12 (docstring vs implemented selection rule; cross-org duplicate rows in Table 3):**
  FIXED (docstring matches the implemented rule; the table subset is additionally collapsed
  by geometry signature, removing the three identical Mixtral-8x22B rows).
- **n1 (geometry R^2 near-tautological):** ADDRESSED (explicit caveat sentence).
- **n2 (PDF not byte-reproducible):** noted; only `make analysis` determinism is required
  by the plan/gate and it holds.
- **n3 (float32 vs BF16 in validation):** ADDRESSED (sentence added).
- **n4 (`+legacycfg` labels unexplained):** ADDRESSED (Methods).
- **n5 (abstract phrasing):** FIXED (validation sentence disentangled; "every public model
  config" softened with the 9-parse-failure count).
- **n6 (fig4 renders infeasible cells at floor color):** accepted as cosmetic; the caption
  and table distinguish "--" and "inf".

## claims-auditor

- **MAJOR-1 (Table 1 rows column):** same as m1, FIXED.
- **MAJOR-2 (dedup misdescription):** same as M2, ADDRESSED (description corrected +
  both-unit reporting + robustness macros; PLAN deviation note added).
- **MAJOR-3 (overhead sensitivity grid promised, not reported):** FIXED (full grid CSV +
  Table `tab:sensitivity` + corrected Methods sentence).
- DistServe/CLA/edge-memory citation overreach: REPHRASED to what the retrieved records
  support.
- "No Llama rows at all": CORRECTED (no rows from the gated meta-llama org; Llama-derived
  checkpoints from included publishers are present).
- "Partial correlations" terminology: replaced (standardized coefficients + VIF framing).
- GQA-ratio direction: DEFINED in Methods.
- AI-use disclosure overclaim: softened ("design constants and calendar years appear as
  literals").
- Pre-registered items unreported (IQRs, concurrency, incremental R^2): all now reported.
- Validation-row/class phrasing in abstract and conclusion: disentangled.
- Writer-corrected counts (107 tests, 61 real config snapshots): kept as verified true.
