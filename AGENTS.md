# AGENTS.md — rules for every agent in this project (orchestrator and all subagents)

Read `BRIEF.md` first. It is the research specification.

## Research integrity (non-negotiable)
1. **No invented numbers.** Every quantitative statement in the paper must trace
   to code output. In practice: a macro or table in `paper/generated/`, written
   by `make analysis` or `make validate`. Never type a result number into prose
   by hand.
2. **No invented citations.** Add a reference only after retrieving it from the
   arXiv API, OpenAlex, or Crossref in this project. Store the raw API response
   under `data/raw/literature/`. Record the id (arXiv eprint or DOI) in the bib
   entry. Cite only what the retrieved abstract supports. If you cannot verify
   a source, do not cite it.
3. **Honest reporting.** Report refuted hypotheses, outliers, measurement
   gaps, and the limits of the data (gated repos skipped, createdAt used as the
   release proxy, Hub coverage bias). Do not p-hack. Fix the analysis plan in
   `analysis/PLAN.md` before looking at the trend results.
4. **The gate is the referee.** Never edit, copy, or weaken
   `/home/ozkangu/Desktop/research/_gates/`. Fix the work, not the check.

## Sandbox rules
- Work only inside `/home/ozkangu/Desktop/research/kv-census`. Do not read or
  write anywhere else, except to read the gate script and the local
  Hugging Face cache.
- No `sudo`, no global `pip install`, and no changes to system or user config.
  Use `uv` with the project-local `.venv` (`uv venv`, `uv pip install ...`)
  and pin versions in `requirements.txt`.
- **CPU only.** Export `CUDA_VISIBLE_DEVICES=""` for every Python process. The
  GPU belongs to other workloads. Never kill or touch processes you did not
  start.
- **Network:** only the public APIs listed below.
  - huggingface.co: ungated, no token.
  - export.arxiv.org: at most 1 request every 3 s.
  - api.openalex.org and api.crossref.org: at most 5 requests/s.
  - Identify requests with a plain User-Agent. Do not send any email address
    or personal data.
  - Cache every response on disk so that re-runs are offline.
- **Git:** use local git only, never push. Commit at the end of every phase
  with a clear message.

## Swarm protocol (the orchestrator is the root session)
- **Delegation.** Use `rlm.spawn(...)` for parallel workstreams, with at most
  **4 subagents running at once** because the model API has rate limits. Give
  each subagent:
  - a self-contained prompt
  - the exact files it owns
  - its acceptance criteria
  - the instruction to read AGENTS.md and BRIEF.md
- **Coordination.** Subagents own disjoint files. Shared interfaces, such as the
  `analysis/kv.py` function signatures and the `data/models.csv` schema, are
  written into `analysis/INTERFACES.md` by the orchestrator before Phase 1
  starts.
- **Messaging.**
  - Subagents report with `rlm.progress_note(...)` and finish with a short
    summary via `agent_message.send(..., receiver_role='parent')`.
  - Collect results with `rlm.collect(..., timeout_ms=...)`.
  - Verify a subagent's output yourself (run its tests or scripts) before
    accepting it.
- **Progress log.** After every phase, append to `PROGRESS.md`, in Turkish: what
  was done, the key numbers (taken from generated files), open issues, and the
  next step.

### Phases
0. **Setup (orchestrator).** git init and commit the brief, then create:
   - the skeleton and Makefile
   - `uv` venv and `requirements.txt`
   - `analysis/INTERFACES.md`
   - `analysis/PLAN.md` (pre-registered analysis plan)
1. **Parallel build (≤ 4 subagents).**
   - `census-data`: model enumeration, config snapshots, metadata, `data/models.csv` builder.
   - `kv-accounting`: `analysis/kv.py` covering every attention class, with unit tests.
   - `validation`: CPU experiment script and `make validate`.
   - `literature`: verified related work, `paper/references.bib`, and annotated notes in `notes/related_work.md`.
2. **Analysis (orchestrator or 1–2 subagents).**
   - RQ1–RQ3 statistics, figures, and `paper/generated/*.tex`.
   - Deterministic `make analysis`.
   - Run `make validate` for RQ4.
3. **Writing (≤ 4 subagents in parallel, one section file each under `paper/sections/`).**
   - Writers use only macros from `paper/generated/` for numbers, and only verified bib keys.
   - The orchestrator then integrates the sections and unifies the voice.
4. **Adversarial review (2 fresh subagents).**
   - `methods-reviewer`: statistics, confounds, whether the conclusions follow from the figures.
   - `claims-auditor`: every sentence that contains a number or a citation is checked against the generated files and the retrieved abstracts.
   - Both write `reviews/*.md`. Fix every major issue and record the responses in `reviews/responses.md`.
5. **Gate.**
   - Run the completion gate (see BRIEF.md) and iterate until it passes.
   - Write `SUMMARY_TR.md`.
   - Make the final commit.
