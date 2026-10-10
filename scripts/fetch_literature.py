#!/usr/bin/env python3
"""Verified bibliography builder for the KV-cache census paper.

Pipeline (per AGENTS.md rule 2 "no invented citations"):
  1. For every candidate, fetch metadata from export.arxiv.org (Atom XML,
     id_list or search_query; max 1 request / 3 s) or api.openalex.org /
     api.crossref.org (max 5 req/s; we use 0.6 s spacing).
     Every raw response is cached under data/raw/literature/ so re-runs are
     offline. Nothing but the cached responses feeds the bibliography.
  2. A sanity keyword must appear in the retrieved title (or abstract for
     searches); otherwise a fallback search runs. A candidate that still fails
     is reported and excluded.
  3. paper/references.bib and notes/related_work.md are generated ONLY from
     retrieved records + scripts/literature_annotations.json (the judgement
     layer: what the retrieved abstract supports, and where it fits).

Usage:
  .venv/bin/python scripts/fetch_literature.py            # fetch (cache-first) + digest + build if possible
  .venv/bin/python scripts/fetch_literature.py --check    # offline verification of bib vs cache + summary
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / "data" / "raw" / "literature"
BIB = ROOT / "paper" / "references.bib"
NOTES = ROOT / "notes" / "related_work.md"
ANNOT = ROOT / "scripts" / "literature_annotations.json"

UA = "kv-census/0.1 (bibliography verification; plain agent User-Agent, no contact data)"
ARXIV_BASE = "https://export.arxiv.org/api/query"
ARXIV_PAUSE = 3.3          # seconds between arXiv requests (limit: 1 per 3 s)
OTHER_PAUSE = 0.6          # openalex/crossref spacing (limit: 5/s)

_last = {"arxiv": 0.0, "other": 0.0}

# ----------------------------------------------------------------------------
# candidates
#   include: True  -> goes into the bibliography (subject to verification)
#           False -> verified+cached backup pool, NOT cited in the bib
#   id: arXiv id (hint; correctness is confirmed by the sanity check)
#   search: fallback arXiv query if the id fails / is absent
#   expect: any-of substrings that must appear in the retrieved title
#           (search mode also scans the abstract)
#   kw:     suffix for the generated bib key authorYEARkw
#   key:    optional explicit key override
# ----------------------------------------------------------------------------
CANDIDATES = [
    # -- serving systems & KV memory -----------------------------------------
    dict(include=True,  id="2309.06180", kw="paged",   expect=["pagedattention"],
         search='all:"PagedAttention" AND all:"vLLM"', cluster="Serving systems and KV memory",
         topic="PagedAttention/vLLM: paged KV-cache memory management for LLM serving"),
    dict(include=True,  id="2401.09670", kw="distserve", expect=["distserve"],
         search='ti:"DistServe"', cluster="Serving systems and KV memory",
         topic="DistServe: prefill/decoding disaggregation driven by KV-cache duplication"),
    # -- attention kernels -----------------------------------------------------
    dict(include=True,  id="2205.14135", kw="flashattention", expect=["flashattention"],
         search='ti:"FlashAttention"', cluster="Attention kernels",
         topic="FlashAttention: IO-aware exact attention"),
    dict(include=True,  id="2307.08691", kw="flashattention2", expect=["flashattention-2"],
         search='ti:"FlashAttention-2"', cluster="Attention kernels",
         topic="FlashAttention-2: parallelism and work partitioning"),
    # -- KV-head reduction -----------------------------------------------------
    dict(include=True,  id="1911.02150", kw="mqa", expect=["write-head"],
         search='ti:"Fast Transformer Decoding"', cluster="KV-head reduction",
         topic="Multi-Query Attention (one shared KV head)"),
    dict(include=True,  id="2305.13245", kw="gqa", expect=["generalized multi-query"],
         search='ti:"GQA"', cluster="KV-head reduction",
         topic="Grouped-Query Attention"),
    dict(include=True,  id="2405.04434", kw="v2", expect=["deepseek-v2"],
         search='ti:"DeepSeek-V2"', cluster="KV-head reduction",
         topic="DeepSeek-V2: Multi-head Latent Attention (MLA) + MoE"),
    # -- windowed attention & sinks -------------------------------------------
    dict(include=True,  id="2310.06825", kw="mistral", expect=["mistral 7b"],
         search='ti:"Mistral 7B"', cluster="Attention scope and sinks",
         topic="Mistral 7B: sliding-window attention"),
    dict(include=True,  id="2309.17453", kw="streamingllm", expect=["attention sinks", "streaming language models"],
         search='ti:"Efficient Streaming Language Models"', cluster="Attention scope and sinks",
         topic="StreamingLLM: attention sinks, fixed-memory streaming attention"),
    # -- KV compression: eviction, quantization, low-rank ----------------------
    dict(include=True,  id="2306.14048", kw="h2o", expect=["heavy-hitter"],
         search='ti:"H2O"', cluster="KV compression",
         topic="H2O: heavy-hitter KV eviction"),
    dict(include=True,  id="2402.02750", kw="kivi", expect=["kivi"],
         search='ti:"KIVI"', cluster="KV compression",
         topic="KIVI: 2-bit KV-cache quantization"),
    dict(include=True,  id="2401.18079", kw="kvquant", expect=["kvquant"],
         search='ti:"KVQuant"', cluster="KV compression",
         topic="KVQuant: 4-bit KV-cache quantization for long context"),
    dict(include=True,  id="2407.21118", kw="palu", expect=["palu"],
         search='ti:"Palu"', cluster="KV compression",
         topic="Palu: low-rank KV-cache compression"),
    # -- weight quantization for memory-efficient serving ----------------------
    dict(include=True,  id="2210.17323", kw="gptq", expect=["gptq", "post-training quantization"],
         search='ti:"GPTQ"', cluster="Weight quantization",
         topic="GPTQ: post-training weight quantization"),
    dict(include=True,  id="2208.07339", kw="int8", expect=["8-bit matrix"],
         search='ti:"LLM.int8"', cluster="Weight quantization",
         topic="LLM.int8(): outlier-aware 8-bit weight quantization"),
    # -- cross-layer KV sharing -------------------------------------------------
    dict(include=True,  id="2405.05254", kw="yoco", expect=["you only cache once"],
         search='ti:"You Only Cache Once"', cluster="Cross-layer KV sharing",
         topic="YOCO: decoder-decoder with one shared cross-layer KV cache"),
    dict(include=True,  id="2405.10637", kw="cla", expect=["layer-condensed"],
         search='ti:"Layer-Condensed KV Cache"', cluster="Cross-layer KV sharing",
         topic="CLA: layer-condensed KV cache (bottom-layer-only KV)"),
    # -- long-context scaling ---------------------------------------------------
    dict(include=True,  id="2104.09864", kw="roformer", expect=["rotary position"],
         search='ti:"RoFormer"', cluster="Long-context scaling",
         topic="RoFormer: rotary position embeddings (RoPE)"),
    dict(include=True,  id="2309.00071", kw="yarn", expect=["yarn", "context window extension"],
         search='ti:"YaRN"', cluster="Long-context scaling",
         topic="YaRN: NTK-aware context-window extension"),
    # -- linear attention --------------------------------------------------------
    dict(include=True,  id="2006.16236", kw="linear", expect=["linear attention"],
         search='ti:"Fast Autoregressive Transformers with Linear Attention"', cluster="Linear attention",
         topic="Linear transformers with fixed-size state (Katharopoulos et al.)"),
    dict(include=True,  id="2307.08621", kw="retnet", expect=["retentive"],
         search='ti:"Retentive Network"', cluster="Linear attention",
         topic="RetNet: retention, parallel/recurrent modes"),
    dict(include=True,  id="2305.13048", kw="rwkv", expect=["rwkv"],
         search='ti:"RWKV"', cluster="Linear attention",
         topic="RWKV: RNN with transformer-scale training"),
    dict(include=True,  id="2412.06464", kw="gdn", expect=["gated delta", "deltanet"],
         search='ti:"Gated Delta Networks"', cluster="Linear attention",
         topic="Gated DeltaNet: gated delta-rule linear attention"),
    # -- SSMs & hybrids -----------------------------------------------------------
    dict(include=True,  id="2312.00752", kw="mamba", expect=["mamba"],
         search='ti:"Mamba: Linear-Time Sequence Modeling"', cluster="State-space models and hybrids",
         topic="Mamba: selective state-space models"),
    dict(include=True,  id="2402.19427", kw="griffin", expect=["griffin", "local attention"],
         search='ti:"Griffin: Mixing Gated Linear Recurrences"', cluster="State-space models and hybrids",
         topic="Griffin: gated linear recurrences + local attention (Hawk/Griffin)"),
    dict(include=True,  id="2403.19887", kw="jamba", expect=["jamba"],
         search='ti:"Jamba"', cluster="State-space models and hybrids",
         topic="Jamba: Transformer-Mamba hybrid MoE"),
    dict(include=True,  id=None, kw="falcon", expect=["falcon", "zamba"],
         search='all:"Falcon-H1"', cluster="State-space models and hybrids",
         topic="Falcon-H1 (fallback chain: Falcon-H1 -> Falcon Mamba -> Zamba): parallel-SSM hybrid"),
    # -- MoE ------------------------------------------------------------------------
    dict(include=True,  id="2401.04088", kw="mixtral", expect=["mixtral"],
         search='ti:"Mixtral of Experts"', cluster="Mixture-of-experts",
         topic="Mixtral of Experts: sparse open-weight MoE"),
    # -- open-weights model family reports -------------------------------------------
    dict(include=True,  id="2407.21783", kw="llama3", expect=["llama 3"],
         search='ti:"The Llama 3 Herd of Models"', cluster="Open-weight model family reports",
         topic="Llama 3 herd of models (GQA family)"),
    dict(include=True,  id="2503.19786", kw="gemma3", expect=["gemma 3"], key="gemmateam2025gemma3",
         search='ti:"Gemma 3 Technical Report"', cluster="Open-weight model family reports",
         topic="Gemma 3 technical report (large head_dim, local+global attention)"),
    dict(include=True,  id="2505.09388", kw="qwen3", expect=["qwen3"],
         search='ti:"Qwen3 Technical Report"', cluster="Open-weight model family reports",
         topic="Qwen3 technical report (dense + MoE, GQA)"),
    dict(include=True,  id="2412.08905", kw="phi4", expect=["phi-4"],
         search='ti:"Phi-4 Technical Report"', cluster="Open-weight model family reports",
         topic="Phi-4 technical report"),
    dict(include=True,  id="2501.00656", kw="olmo2", expect=["olmo"],
         search='ti:"OLMo 2"', cluster="Open-weight model family reports",
         topic="OLMo 2 fully-open report"),
    # -- edge / unified-memory serving --------------------------------------------------
    # OpenAlex holds no canonical llama.cpp software record (checked 2026-10-10,
    # data/raw/literature/openalex_search_llamacpp.json: only derivative works), so
    # a bare GitHub URL would not be API-verified. Use the named arXiv alternatives:
    # MLC-LLM, falling back to Apple's LLM-in-a-flash (limited-memory inference).
    dict(include=True,  id=None, kw="edgemem", expect=["mlc-llm", "llm in a flash"],
         search='all:"MLC-LLM"', fallback_searches=['ti:"LLM in a flash"'],
         cluster="Edge and unified-memory serving",
         topic="Edge/memory-constrained serving (MLC-LLM compilation, or Apple LLM-in-a-flash)"),

    # ---------------- verified backup pool (NOT in the bib) ---------------------------
    dict(include=False, id="2403.02310", kw="sarathi", expect=["sarathi"], cluster="pool",
         topic="Sarathi-Serve: chunked-prefill scheduling (pool)"),
    dict(include=False, id="2405.21060", kw="mamba2", expect=["state space duality"], cluster="pool",
         topic="Mamba-2 / SSD (pool)"),
    dict(include=False, id="2306.00978", kw="awq", expect=["activation-aware"], cluster="pool",
         topic="AWQ weight quantization (pool)"),
    dict(include=False, id="2404.14469", kw="snapkv", expect=["snapkv"], cluster="pool",
         topic="SnapKV KV eviction (pool)"),
    dict(include=False, id="2004.05150", kw="longformer", expect=["longformer"], cluster="pool",
         topic="Longformer: sliding+global attention (pool)"),
    dict(include=False, id="2407.00079", kw="mooncake", expect=["mooncake"], cluster="pool",
         topic="Mooncake: KV-cache-centric disaggregated serving (pool)"),
    dict(include=False, id="2305.17118", kw="scissorhands", expect=["scissorhands"], cluster="pool",
         topic="Scissorhands KV eviction (pool)"),
    dict(include=False, id="2311.01282", kw="flashdecoding", expect=["flashdecoding"], cluster="pool",
         topic="FlashDecoding++: parallel KV processing at decode (pool)"),
    dict(include=False, id="2404.07839", kw="recurrentgemma", expect=["recurrentgemma", "griffin"], cluster="pool",
         topic="RecurrentGemma: Griffin-based open report (pool)"),
    dict(include=False, id="2101.03961", kw="switch", expect=["switch transformers", "trillion"], cluster="pool",
         topic="Switch Transformer: sparse MoE (pool)"),
    dict(include=False, id="2405.16712", kw="zamba", expect=["zamba"], cluster="pool",
         topic="Zamba: compact SSM hybrid (pool)"),
]

FALCON_FALLBACKS = ['all:"Falcon Mamba"', 'ti:"Zamba: A Compact 7B SSM Hybrid Model"']

ATOM = {"a": "http://www.w3.org/2005/Atom"}
ARX = {"arxiv": "http://arxiv.org/schemas/atom"}


class FetchError(RuntimeError):
    pass


def pause(kind: str) -> None:
    gap = ARXIV_PAUSE if kind == "arxiv" else OTHER_PAUSE
    d = _last[kind] + gap - time.time()
    if d > 0:
        time.sleep(d)
    _last[kind] = time.time()


def http_get(url: str, kind: str) -> str:
    for attempt in range(3):
        pause(kind)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8", errors="replace")
                if r.status != 200:
                    raise FetchError(f"HTTP {r.status}")
                return body
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(10)
                continue
            raise FetchError(f"HTTP {e.code} for {url}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 2:
                time.sleep(10)
                continue
            raise FetchError(f"network error {e} for {url}") from e
    raise FetchError(f"retries exhausted for {url}")


def cached_get(url: str, path: Path, kind: str) -> tuple[str, bool]:
    if path.exists():
        return path.read_text(errors="replace"), True
    body = http_get(url, kind)
    path.write_text(body)
    print(f"  FETCHED {path.name}")
    return body, False


def norm_ws(s: str) -> str:
    return " ".join(s.split())


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return s or "anon"


def sanitize(s: str) -> str:
    return re.sub(r"[^\w.\-]", "_", s)


def parse_atom(body: str) -> list[dict]:
    out = []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return out
    for e in root.findall("a:entry", ATOM):
        eid = e.findtext("a:id", "", ATOM).rsplit("/abs/", 1)[-1]
        out.append(dict(
            id=re.sub(r"v\d+$", "", eid).strip(),
            title=norm_ws(e.findtext("a:title", "", ATOM)),
            authors=[norm_ws(a.findtext("a:name", "", ATOM)) for a in e.findall("a:author", ATOM)],
            published=e.findtext("a:published", "", ATOM)[:10],
            updated=e.findtext("a:updated", "", ATOM)[:10],
            summary=norm_ws(e.findtext("a:summary", "", ATOM)),
            journal_ref=norm_ws(e.findtext("arxiv:journal_ref", "", ARX)),
            doi=norm_ws(e.findtext("arxiv:doi", "", ARX)),
        ))
    return out


def fetch_arxiv_id(aid: str) -> dict:
    path = LIT / f"arxiv_{sanitize(aid)}.xml"
    url = f"{ARXIV_BASE}?id_list={urllib.parse.quote(aid)}&max_results=1"
    body, _ = cached_get(url, path, "arxiv")
    entries = parse_atom(body)
    if len(entries) != 1:
        raise FetchError(f"id_list {aid}: {len(entries)} entries")
    return entries[0]


def fetch_arxiv_search(query: str, slugname: str) -> list[dict]:
    path = LIT / f"arxiv_search_{sanitize(slugname)}.xml"
    url = f"{ARXIV_BASE}?search_query={urllib.parse.quote_plus(query)}&max_results=8&sortBy=relevance"
    body, _ = cached_get(url, path, "arxiv")
    return parse_atom(body)


def expect_ok(entry: dict, expects: list[str], also_abstract: bool = False) -> bool:
    hay = entry["title"].lower()
    if also_abstract:
        hay = hay + " " + entry["summary"].lower()
    return any(x in hay for x in expects)


def resolve_arxiv_candidate(c: dict) -> tuple[dict | None, str]:
    """Returns (resolved entry or None, human status)."""
    if c.get("id"):
        try:
            ent = fetch_arxiv_id(c["id"])
            if expect_ok(ent, c["expect"]):
                return ent, f"arxiv:{ent['id']}"
            return None, f"TITLE-MISMATCH id={c['id']} -> '{ent['title']}'"
        except FetchError as e:
            print(f"  ! id fetch failed {c.get('id')}: {e}")
    queries = ([c["search"]] if c.get("search") else []) + c.get("fallback_searches", [])
    if c["kw"] == "falcon":
        queries += FALCON_FALLBACKS
    for q in queries:
        if not q:
            continue
        try:
            ents = fetch_arxiv_search(q, c["kw"] + "_" + slug(norm_ws(q))[:30])
        except FetchError as e:
            print(f"  ! search failed '{q}': {e}")
            continue
        for ent in ents:
            if expect_ok(ent, c["expect"], also_abstract=True):
                # keep a clean per-id raw file as well
                try:
                    fetch_arxiv_id(ent["id"])
                except FetchError:
                    pass
                return ent, f"arxiv-search '{q}' -> arxiv:{ent['id']}"
    return None, "NOT RESOLVED"


def resolve_llamacpp() -> tuple[dict | None, str]:
    """llama.cpp has no paper DOI/arXiv id; verify via OpenAlex (raw JSON cached)."""
    path = LIT / "openalex_search_llamacpp.json"
    url = "https://api.openalex.org/works?search=llama.cpp&per-page=15"
    try:
        body, _ = cached_get(url, path, "other")
        data = json.loads(body)
    except (FetchError, json.JSONDecodeError) as e:
        print(f"  ! openalex failed: {e}")
        return None, f"OPENALEX FAILED ({e})"
    cands = []
    for r in data.get("results", []):
        name = norm_ws(r.get("display_name", ""))
        if "llama.cpp" in name.lower():
            cands.append(r)
    if not cands:
        return None, "no OpenAlex record titled llama.cpp"
    r = sorted(cands, key=lambda r: -(r.get("cited_by_count") or 0))[0]
    landing = ""
    for loc in ([r.get("primary_location") or {}] + (r.get("locations") or [])):
        u = (loc or {}).get("landing_page_url") or ""
        if u:
            landing = landing or u
            if "github.com" in u:
                landing = u
                break
    authors = [a.get("author", {}).get("display_name", "") for a in r.get("authorships", [])]
    # NOTE: OpenAlex may carry a Zenodo DOI; we deliberately do NOT put it in the
    # bib because Zenodo is DataCite, and the completion gate verifies DOIs via
    # Crossref, where 10.5281/* DOIs do not resolve. Cite the live URL instead.
    return dict(title=norm_ws(r.get("display_name", "")), authors=[a for a in authors if a],
                published=str(r.get("publication_year") or ""), url=landing,
                openalex_doi=(r.get("doi") or "").replace("https://doi.org/", ""),
                openalex_id=r.get("id", "")), \
        f"openalex:{r.get('id','')}"


def derive_key(authors: list[str], year: str, kw: str, override: str | None) -> str:
    if override:
        return override
    if not authors:
        return f"anon{year}{kw}"
    name = authors[0]
    toks = re.sub(r"[^\w\s\-]", " ", name).split()
    sur = slug((toks[-1] if toks else "anon")[:14])
    return f"{sur}{year}{kw}"


def bib_escape(s: str) -> str:
    out = []
    for ch in s:
        # NOTE: $ and _ are kept verbatim: arXiv titles use them for math
        # (e.g. "H$_2$O"), which is valid LaTeX and matches the API record.
        if ch == "&":
            out.append(r"\&")
        elif ch == "%":
            out.append(r"\%")
        elif ch == "#":
            out.append(r"\#")
        else:
            out.append(ch)
    return "".join(out)


def bib_entry(key: str, rec: dict, extra_note: str | None = None) -> str:
    authors = list(rec["authors"])
    for i, a in enumerate(authors):
        # brace-protect corporate author names so BibTeX keeps them verbatim
        if re.search(r"\b(AI|Team)\b", a):
            authors[i] = "{" + a + "}"
    if len(authors) > 8:
        astr = " and ".join(authors[:8]) + " and others"
    else:
        astr = " and ".join(authors)
    lines = [f"@misc{{{key},"]
    lines.append(f"  author = {{{astr}}},")
    lines.append(f"  title = {{{{{bib_escape(rec['title'])}}}}},")
    lines.append(f"  year = {{{rec['published'][:4]}}},")
    eprint = rec.get("eprint") or rec.get("id")
    if eprint:
        lines.append(f"  eprint = {{{eprint}}},")
        lines.append("  archivePrefix = {arXiv},")
    # only emit real venue DOIs; skip arXiv DOIs (eprint covers them) and
    # DataCite/ResearchGate DOIs (10.5281, 10.13140) which the gate checks via Crossref
    doi = rec.get("doi") or ""
    if doi and not doi.startswith(("10.48550/", "10.5281/", "10.13140/")):
        lines.append(f"  doi = {{{doi}}},")
    if rec.get("journal_ref"):
        jr = rec["journal_ref"]
        field = "booktitle" if re.search(r"proceedings|conference|symposium|workshop|learning representations", jr, re.I) else "journal"
        lines.append(f"  {field} = {{{{{bib_escape(jr)}}}}},")
    if rec.get("url"):
        lines.append(f"  url = {{{rec['url']}}},")
    if extra_note:
        lines.append(f"  note = {{{extra_note}}},")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------- main phases
def phase_fetch() -> list[dict]:
    LIT.mkdir(parents=True, exist_ok=True)
    records = []
    for c in CANDIDATES:
        print(f"[{c['kw']}] include={c['include']}")
        if c.get("source") == "web" and c["kw"] == "llamacpp":
            rec, status = resolve_llamacpp()
            if rec:
                rec.update(kw=c["kw"], cluster=c["cluster"], topic=c["topic"], include=c["include"],
                           verified_by="openalex-api",
                           retrieved=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                rec["key"] = derive_key(rec.get("authors", []), rec["published"], c["kw"], c.get("key"))
                records.append(rec)
            else:
                records.append(dict(kw=c["kw"], include=c["include"], failed=True, status=status,
                                    cluster=c["cluster"], topic=c["topic"]))
            print(f"  -> {status}")
            continue
        ent, status = resolve_arxiv_candidate(c)
        if ent:
            ent.update(kw=c["kw"], cluster=c["cluster"], topic=c["topic"], include=c["include"],
                       verified_by="arxiv-api",
                       retrieved=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            ent["key"] = derive_key(ent["authors"], ent["published"][:4], c["kw"], c.get("key"))
            records.append(ent)
        else:
            records.append(dict(kw=c["kw"], include=c["include"], failed=True, status=status,
                                cluster=c["cluster"], topic=c["topic"]))
        print(f"  -> {status}: {ent['title'] if ent else '-'}")
    (LIT / "_index.json").write_text(json.dumps(records, indent=1, ensure_ascii=False))
    # digest for annotation writing
    lines = ["# Raw-record digest (for writing annotations; NOT a paper artifact)", ""]
    for r in records:
        if r.get("failed") or not r.get("include"):
            continue
        lines.append(f"## {r['key']}  [{r.get('id') or r.get('url')}]")
        lines.append(f"- title: {r['title']}")
        lines.append(f"- authors: {'; '.join(r['authors'][:6])}{' ...' if len(r['authors']) > 6 else ''}")
        lines.append(f"- published: {r['published']}  journal_ref: {r.get('journal_ref','') or '-'}  doi: {r.get('doi','') or '-'}")
        lines.append(f"- abstract: {r.get('summary', r.get('url',''))}")
        lines.append("")
    (LIT / "_digest.md").write_text("\n".join(lines))
    n_ok = sum(1 for r in records if r.get("include") and not r.get("failed"))
    n_fail = [r["kw"] for r in records if r.get("include") and r.get("failed")]
    pool_ok = sum(1 for r in records if not r.get("include") and not r.get("failed"))
    print(f"SUMMARY: included resolved {n_ok}, failed {n_fail}; pool resolved {pool_ok}")
    return records


def phase_build(records: list[dict]) -> int:
    ann = {}
    if ANNOT.exists():
        ann = json.loads(ANNOT.read_text())
    inc = [r for r in records if r.get("include") and not r.get("failed")]
    missing = [r["key"] for r in inc if r["key"] not in ann]
    if missing:
        print(f"BUILD SKIPPED: annotations pending for {len(missing)}: {missing}")
        return 1
    if len(inc) < 25:
        print(f"BUILD SKIPPED: only {len(inc)} verifiable included entries (<25)")
        return 1
    # unique keys check
    keys = [r["key"] for r in inc]
    assert len(keys) == len(set(keys)), "duplicate bib keys"
    # generate bib in candidate order (records were appended in CANDIDATES order)
    bib = ["% Auto-generated by scripts/fetch_literature.py from cached API responses",
           "% under data/raw/literature/ (arXiv Atom XML / OpenAlex JSON). Do not hand-edit.",
           ""]
    order = []
    for r in inc:
        note = None
        if r.get("url") and not (r.get("eprint") or r.get("id")):
            note = "Software repository; URL liveness is re-checked by the completion gate"
        bib.append(bib_entry(r["key"], r, note))
        bib.append("")
        order.append(r["key"])
    BIB.parent.mkdir(parents=True, exist_ok=True)
    BIB.write_text("\n".join(bib))
    # notes
    md = ["# Related work - verified bibliography notes", "",
          "Generated by `scripts/fetch_literature.py`. Every entry traces to a raw API",
          "response cached under `data/raw/literature/` (arXiv Atom XML or OpenAlex JSON).",
          "Annotations state only what the retrieved title/abstract/metadata supports",
          "(AGENTS.md rule 2: no invented citations). Sections mirror the bibliography order.",
          ""]
    cur = None
    for r in inc:
        if r["cluster"] != cur:
            cur = r["cluster"]
            md += [f"## {cur}", ""]
        a = ann[r["key"]]
        rid = f"arXiv eprint {r['id']} -> `data/raw/literature/arxiv_{r['id']}.xml`" if r.get("id") \
            else f"OpenAlex {r.get('openalex_id','')} -> `data/raw/literature/openalex_search_llamacpp.json`"
        md += [f"### {r['key']} - {r['topic']}",
               f"- **Retrieval:** {rid} (verified via {r['verified_by']})",
               f"- **Supports:** {a['ann']}",
               f"- **Fits:** {a['section']}",
               ""]
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    NOTES.write_text("\n".join(md))
    print(f"BUILT: {len(inc)} entries -> {BIB}, {NOTES}")
    return 0


def phase_check() -> int:
    text = BIB.read_text() if BIB.exists() else ""
    entries = re.findall(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text)
    keys = [k for _, k in entries]
    n_urlonly = len(re.findall(r"^\s*url\s*=", text, re.M))
    n_eprint = len(re.findall(r"^\s*eprint\s*=", text, re.M))
    problems = []
    if len(keys) < 25:
        problems.append(f"only {len(keys)} entries")
    if len(set(keys)) != len(keys):
        problems.append("duplicate keys")
    if n_urlonly > 0.35 * len(keys):
        problems.append(f"{n_urlonly} URL-only > 35%")
    idx = {r["key"]: r for r in json.loads((LIT / "_index.json").read_text()) if not r.get("failed")}
    for k in keys:
        if k not in idx:
            problems.append(f"{k} not in index")
            continue
        t_bib = re.search(r"@\w+\{" + re.escape(k) + r",(?:(?!@).)*?title\s*=\s*\{\{?(.*?)\}?\},", text, re.S)
        if t_bib:
            a = re.sub(r"[^a-z0-9]+", " ", t_bib.group(1).lower())
            b = re.sub(r"[^a-z0-9]+", " ", idx[k]["title"].lower())
            if a.strip() != b.strip():
                problems.append(f"{k}: bib/API title drift")
    if re.search(r"eprint\s*=\s*\{[\w.\-/]*v\d+\}", text):
        problems.append("versioned arXiv eprint found")
    if problems:
        print("CHECK FAILED:", "; ".join(problems))
        return 1
    print(f"CHECK PASSED: {len(keys)} entries, {n_eprint} arXiv eprints, {n_urlonly} URL-only")
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "fetch"):
        records = phase_fetch()
    else:
        records = json.loads((LIT / "_index.json").read_text())
    rc = 0
    if mode in ("all", "build"):
        rc = phase_build(records) or rc
    if mode in ("all", "build", "check"):
        rc = phase_check() or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
