#!/usr/bin/env python3
"""Stage B: build data/models.csv from the Stage-A snapshots.

Reads only cached artifacts (offline):
  data/raw/census_manifest.json  Stage-A decisions (included repos)
  data/raw/configs/<org>__<name>.json  raw configs
  data/gated_skipped.txt         gated limitation count (for the report)

Writes:
  data/models.csv                census table, EXACT schema of INTERFACES.md s.2
  data/raw/census_csv_report.json  verification report (offline, no timestamps)

All KV/geometry values come from analysis/kv.py (imported, never re-implemented).
Total params prefer the Hub safetensors metadata (params_source=safetensors);
otherwise a config-based formula for standard SwiGLU decoder families
(params_source=config, formula below). active_params_b equals total_params_b
for dense models; for MoE it is the standard approximation
  active = total - n_moe_layers * (E - K) * 3 * moe_intermediate_size * H
(only non-shared routed experts are deactivated; router + shared expert stay).
Rows whose architecture kv.py cannot parse are recorded in
data/raw/kv_errors.json and skipped (count reported).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import kv  # noqa: E402  (analysis/kv.py, owner: kv-accounting)

MANIFEST = ROOT / "data" / "raw" / "census_manifest.json"
CONFIGS = ROOT / "data" / "raw" / "configs"
GATED_TXT = ROOT / "data" / "gated_skipped.txt"
OUT_CSV = ROOT / "data" / "models.csv"
REPORT = ROOT / "data" / "raw" / "census_csv_report.json"
KV_ERRORS = ROOT / "data" / "raw" / "kv_errors.json"

COLUMNS = [
    "repo_id", "org", "model_name", "release_date", "params_source",
    "total_params_b", "active_params_b", "moe", "num_layers",
    "num_attention_heads", "num_kv_heads", "head_dim", "hidden_size",
    "attention_type", "gqa_ratio", "window_size", "n_full_layers",
    "n_window_layers", "n_latent_layers", "n_recurrent_layers",
    "kv_shared_cross_layer", "kv_lora_rank", "qk_rope_head_dim",
    "recurrent_state_bytes", "kv_bytes_per_token_bf16", "kv_bytes_at_32k",
    "kv_bytes_at_128k", "dedup_group", "notes",
]

DESCRIBE_TO_COLUMN = {
    "num_layers": "num_layers",
    "num_attention_heads": "num_attention_heads",
    "num_kv_heads": "num_kv_heads",
    "head_dim": "head_dim",
    "hidden_size": "hidden_size",
    "attention_type": "attention_type",
    "gqa_ratio": "gqa_ratio",
    "window_size": "window_size",
    "n_full_layers": "n_full_layers",
    "n_window_layers": "n_window_layers",
    "n_latent_layers": "n_latent_layers",
    "n_recurrent_layers": "n_recurrent_layers",
    "kv_shared_cross_layer": "kv_shared_cross_layer",
    "kv_lora_rank": "kv_lora_rank",
    "qk_rope_head_dim": "qk_rope_head_dim",
    "recurrent_state_bytes": "recurrent_state_bytes",
    "kv_bytes_per_token_bf16": "kv_bytes_per_token_bf16",
    "kv_bytes_at_32k": "kv_bytes_at_32k",
    "kv_bytes_at_128k": "kv_bytes_at_128k",
}

# dedup: trailing name tokens stripped (case-insensitive) + pure date-ish digits
DEDUP_SUFFIX_TOKENS = {
    "instruct", "chat", "base", "it", "thinking", "reasoning", "latest",
    "sft", "preview",
}
_DATEISH = re.compile(r"\d{3,6}")

# MoE detection: config keys first, model_type fallback
MOE_EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts",
                   "num_routed_experts")
MOE_MODEL_TYPES = {
    "mixtral", "qwen2_moe", "qwen3_moe", "qwen3_next", "dbrx", "arctic",
    "granite_moe", "granite_moe_shared", "olmoe", "phimoe", "jamba",
    "deepseek_v2", "deepseek_v3", "deepseek_v4", "deepseek_v32",
    "deepseek_v41", "glm4_moe", "kimi_k2", "minimax",
    "hunyuan", "bailing_moe", "llama4",
}

# config-based param formula coverage (params_source=config)
SWIGLU_TYPES = {
    "llama", "mistral", "mixtral", "qwen2", "qwen3", "yi", "gemma", "gemma2",
    "gemma3_text", "gemma3", "phi", "phi3", "falcon3", "internlm2",
    "baichuan", "granite", "stablelm", "stablecode", "olmo", "olmo2",
    "deepseek", "chatglm", "glm4", "exaone", "codegemma", "arctic", "dbrx",
    "starcoder2", "openelm", "mpt", "nemotron", "olmoe", "hunyuan",
    "hunyuan_v1_dense",
}


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


def scope_text(config: dict) -> dict:
    """Pick the language-backbone sub-dict (CHECK paths only; KV math is kv.py's).
    Walks the known nesting keys recursively (e.g. Qwen-Omni nests the text
    backbone under thinker_config.text_config) until a dict carrying a layer
    count (num_hidden_layers / n_layer / num_layers) is found."""
    nest = ("text_config", "llm_config", "language_config", "text_model",
            "thinker_config")
    stack, seen = [config], set()
    while stack:
        cur = stack.pop(0)
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if not isinstance(cur, dict):
            continue
        if any(k in cur for k in ("num_hidden_layers", "n_layer", "num_layers")):
            # prefer the deepest text scope: keep descending if a nested
            # text_config also carries a layer count
            deeper = cur.get("text_config")
            if isinstance(deeper, dict) and any(
                    k in deeper for k in ("num_hidden_layers", "n_layer", "num_layers")):
                return scope_text(deeper)
            return cur
        for key in nest:
            sub = cur.get(key)
            if isinstance(sub, dict):
                stack.append(sub)
    return config


def is_moe(cfg_txt: dict, model_type: str) -> bool:
    for k in MOE_EXPERT_KEYS:
        if isinstance(cfg_txt.get(k), int) and cfg_txt[k] > 1:
            return True
    if isinstance(cfg_txt.get("moe_intermediate_size"), int):
        return True
    return model_type in MOE_MODEL_TYPES


def moe_shape(cfg_txt: dict):
    """(E experts/layer, K active, moe_intermediate, n_moe_layers) or None."""
    E = None
    for k in MOE_EXPERT_KEYS:
        if isinstance(cfg_txt.get(k), int) and cfg_txt[k] > 1:
            E = cfg_txt[k]
            break
    if E is None:
        return None
    K = cfg_txt.get("num_experts_per_tok") or cfg_txt.get("num_selected_experts") or cfg_txt.get("n_shared_experts")
    I = cfg_txt.get("moe_intermediate_size") or cfg_txt.get("expert_intermediate_size")
    L = cfg_txt.get("num_hidden_layers") or 0
    pat = cfg_txt.get("layers_block_type") or cfg_txt.get("layer_types") or []
    n_moe = sum(1 for t in pat if "moe" in str(t).lower()) or (L if pat == [] else 0)
    if not pat:
        n_moe = L
    if isinstance(K, int) and isinstance(I, int) and n_moe > 0:
        return E, K, I, n_moe
    return None


def config_params_b(cfg_txt: dict, model_type: str):
    """Total params for standard decoder families (millions of params -> None if unknown)."""
    H = cfg_txt.get("hidden_size")
    V = cfg_txt.get("vocab_size")
    L = cfg_txt.get("num_hidden_layers")
    I = cfg_txt.get("intermediate_size")
    if model_type == "gpt_neox":
        # fused QKV (3H^2) + output (H^2) + 2-matrix gelu MLP; matches the
        # F16/F32 safetensors count of pythia-410m within 0.1%
        if not all(isinstance(x, int) and x > 0 for x in (H, V, L, I)):
            return None
        p = V * H * (0 if cfg_txt.get("tie_word_embeddings") else 2)
        p += L * (4 * H * H + 2 * H * I)
        return p / 1e9
    if model_type not in SWIGLU_TYPES:
        return None
    V = cfg_txt.get("vocab_size")
    L = cfg_txt.get("num_hidden_layers")
    I = cfg_txt.get("intermediate_size")
    nh = cfg_txt.get("num_attention_heads")
    nkv = cfg_txt.get("num_key_value_heads") or nh
    hd = cfg_txt.get("head_dim") or (H // nh if H and nh else None)
    if not all(isinstance(x, int) and x > 0 for x in (H, V, L, I, nh, hd)):
        return None
    p = V * H  # token embedding
    p += 0 if cfg_txt.get("tie_word_embeddings") else V * H
    p += L * H * hd * (2 * nh + 2 * nkv)  # q k v o
    if is_moe(cfg_txt, model_type):
        shape = moe_shape(cfg_txt)
        if shape is None:
            return None
        E, _K, Im, n_moe = shape
        p += n_moe * (E * 3 * Im * H + H * E)  # experts + router
        shared = cfg_txt.get("shared_expert_intermediate_size")
        if isinstance(shared, int) and shared > 0:
            p += n_moe * 3 * shared * H
        dense_mlps = L - n_moe
        p += max(dense_mlps, 0) * 3 * H * I
    else:
        p += L * 3 * H * I
    return p / 1e9


def stem_and_stripped(name: str):
    """Base-name stem with instruct/chat/base(-like) and date-ish suffixes stripped."""
    tokens = re.split(r"[-_]", name)
    stripped = []
    while tokens:
        last = tokens[-1]
        low = last.lower()
        if low in DEDUP_SUFFIX_TOKENS or _DATEISH.fullmatch(last):
            stripped.append(low)
            tokens.pop()
        else:
            break
    return "-".join(tokens), stripped


def geometry_signature(d: dict) -> str:
    keys = [
        "num_layers", "num_attention_heads", "num_kv_heads", "head_dim",
        "hidden_size", "attention_type", "window_size", "n_full_layers",
        "n_window_layers", "n_latent_layers", "n_recurrent_layers",
        "kv_shared_cross_layer", "kv_lora_rank", "qk_rope_head_dim",
        "recurrent_state_bytes", "kv_bytes_per_token_bf16",
        "kv_bytes_at_32k", "kv_bytes_at_128k",
    ]
    payload = json.dumps({k: d.get(k) for k in keys}, sort_keys=True)
    return payload


def build() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text())
    included = [e for e in manifest if e.get("decision") == "included"]
    rows, errors = [], []
    for e in sorted(included, key=lambda x: x["repo_id"]):
        repo_id = e["repo_id"]
        cfg_path = CONFIGS / f"{e['org']}__{e['name']}.json"
        if not cfg_path.exists():
            errors.append({"repo_id": repo_id, "error": "config_file_missing"})
            continue
        cfg = json.loads(cfg_path.read_text())
        try:
            d = kv.describe(cfg)
        except Exception as exc:  # KVConfigError and friends -> recorded, skipped
            errors.append({"repo_id": repo_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        missing = [k for k in DESCRIBE_TO_COLUMN if k not in d]
        if missing:
            errors.append({"repo_id": repo_id, "error": f"describe_missing_keys:{missing}"})
            continue

        cfg_txt = scope_text(cfg)
        model_type = str(cfg_txt.get("model_type") or d.get("arch") or "")
        notes = [f"arch={d.get('arch')}"]
        if e.get("reason") == "multimodal_text":
            notes.append("language_backbone_of_multimodal")
            notes.append("params_include_vision_tower")

        # params: Hub safetensors metadata. Legacy GPT-NeoX-style checkpoints
        # ship causal-mask bool buffers (count L x max_pos^2 elements, reported
        # as U8/BOOL) and I64 metadata; those are not parameters. Native 4-bit
        # models (gpt-oss MXFP4) legitimately report their packed weights as
        # U8, so U8/BOOL are subtracted only when their size matches the
        # L x max_position_embeddings^2 mask pattern (1% tolerance).
        st_params = None
        st_detail = e.get("safetensors_parameters") or {}
        if st_detail:
            Lm = cfg_txt.get("num_hidden_layers")
            Mp = cfg_txt.get("max_position_embeddings")
            mask_elems = Lm * Mp * Mp if (isinstance(Lm, int) and isinstance(Mp, int)) else None
            counted = 0
            for dt, v in st_detail.items():
                if dt == "I64":
                    continue  # metadata buffers, never parameters
                if dt in ("U8", "BOOL") and mask_elems and abs(v - mask_elems) <= 0.01 * mask_elems:
                    continue  # causal-mask buffers
                counted += v
            if counted > 0:
                st_params = counted
        elif isinstance(e.get("safetensors_total"), int) and e["safetensors_total"] > 0:
            st_params = e["safetensors_total"]
        if st_params:
            total_b = st_params / 1e9
            params_source = "safetensors"
            if cfg_txt.get("tie_word_embeddings"):
                notes.append("safetensors_total_excludes_tied_lm_head")
        else:
            total_b = config_params_b(cfg_txt, model_type)
            if total_b is None:
                params_source, total_b = "none", None
                notes.append("params_unavailable_no_safetensors_or_formula")
            else:
                params_source = "config"

        # MoE / active params
        moe = is_moe(cfg_txt, model_type)
        active_b = total_b
        if moe and total_b is not None:
            shape = moe_shape(cfg_txt)
            if shape:
                E, K, I_moe, n_moe = shape
                active_b = total_b - n_moe * (E - K) * 3 * I_moe * (cfg_txt.get("hidden_size") or 0) / 1e9
                notes.append(f"active=total-{n_moe}x({E}-{K})x3x{I_moe}xH")
            else:
                active_b = None
                notes.append("active_params_unknown_moe_shape")

        # dedup
        stem, stripped = stem_and_stripped(e["name"])
        if stripped:
            notes.append("dedup_stripped=" + "+".join(stripped))
        group = hashlib.sha1(
            (e["org"] + "|" + stem + "|" + geometry_signature(d)).encode()
        ).hexdigest()[:12]

        row = {
            "repo_id": repo_id,
            "org": e["org"],
            "model_name": e["name"],
            "release_date": (e.get("created_at") or "")[:10],
            "params_source": params_source,
            "total_params_b": round(total_b, 6) if total_b is not None else None,
            "active_params_b": round(active_b, 6) if active_b is not None else None,
            "moe": moe,
            "dedup_group": group,
            "notes": ";".join(notes),
        }
        for dk, ck in DESCRIBE_TO_COLUMN.items():
            row[ck] = d[dk]
        rows.append(row)
    KV_ERRORS.write_text(json.dumps(sorted(errors, key=lambda x: x["repo_id"]), indent=1))
    return rows


def verify(rows: list[dict]) -> tuple[dict, bool]:
    checks = {}
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        checks[name] = ("PASS" if cond else "FAIL") + (f" ({detail})" if detail else "")
        if not cond:
            ok = False

    check("min_150_rows", len(rows) >= 150, f"{len(rows)} rows")
    ids = [r["repo_id"] for r in rows]
    check("unique_repo_id", len(set(ids)) == len(ids))
    years = Counter((r["release_date"] or "")[:4] for r in rows)
    in_window = {y: n for y, n in years.items() if y in ("2023", "2024", "2025", "2026")}
    check("at_least_3_years_2023_2026", len(in_window) >= 3, json.dumps(in_window, sort_keys=True))
    missing_cfg = [r["repo_id"] for r in rows
                   if not (CONFIGS / f"{r['org']}__{r['model_name']}.json").exists()]
    check("config_file_for_every_row", not missing_cfg, f"missing={missing_cfg[:5]}")

    bad_ident, bad_match, unverifiable = [], [], []
    for r in rows:
        if r["attention_type"] not in ("mha", "gqa", "mqa"):
            continue
        expected = 2 * r["num_layers"] * r["num_kv_heads"] * r["head_dim"] * 2
        if int(round(r["kv_bytes_per_token_bf16"])) != expected:
            bad_ident.append(r["repo_id"])
            continue
        cfg = json.loads((CONFIGS / f"{r['org']}__{r['model_name']}.json").read_text())
        t = scope_text(cfg)
        # per-layer lists / absent keys make the raw-config comparison
        # impossible; such rows are reported as unverifiable instead of failed
        if any(isinstance(t.get(k), list) for k in
               ("num_attention_heads", "num_key_value_heads", "num_hidden_layers")):
            unverifiable.append(r["repo_id"] + ":variable_geometry_lists")
            continue
        # legacy keys: GPT-2 (n_layer/n_head/n_embd), EXAONE (num_layers),
        # OLMo-1 native (n_layers/n_heads/n_kv_heads/d_model)
        L2 = (t.get("num_hidden_layers") or t.get("n_layer")
              or t.get("num_layers") or t.get("n_layers"))
        nh2 = (t.get("num_attention_heads") or t.get("n_head")
               or t.get("n_heads") or t.get("num_heads"))
        # kv-head verification mirrors the completion gate's convention
        # (num_key_value_heads or num_attention_heads), extended with the
        # documented legacy signals (ChatGLM multi_query_*, OLMo/falcon
        # n_kv_heads/num_kv_heads, GPT-BigCode multi_query). When a non-standard
        # signal defines the kv heads the gate skips the check; so do we.
        has_std_kvh = t.get("num_key_value_heads") is not None
        legacy_kvh = any(t.get(k) is not None for k in ("num_kv_heads", "n_kv_heads")) or             (t.get("multi_query_attention") and t.get("multi_query_group_num")) or             bool(t.get("multi_query"))
        if has_std_kvh:
            kvh2 = t["num_key_value_heads"]
        elif t.get("num_kv_heads") is not None or t.get("n_kv_heads") is not None:
            kvh2 = t.get("num_kv_heads") or t.get("n_kv_heads")
        elif t.get("multi_query_attention") and t.get("multi_query_group_num"):
            kvh2 = t["multi_query_group_num"]
        elif t.get("multi_query"):
            kvh2 = 1
        elif t.get("num_attention_heads"):
            kvh2 = t["num_attention_heads"]
        else:
            kvh2 = None
        if kvh2 is None:
            unverifiable.append(r["repo_id"] + ":kvh_absent_in_config")
            continue
        H2 = t.get("hidden_size") or t.get("n_embd") or t.get("d_model")
        hd2 = ((t.get("head_dim") or None) or t.get("kv_channels")
               or (H2 // nh2 if H2 and nh2 else None))
        if (r["num_layers"], r["num_kv_heads"], r["head_dim"]) != (L2, kvh2, hd2):
            if legacy_kvh:
                unverifiable.append(r["repo_id"] + ":legacy_kv_key")
            else:
                bad_match.append(r["repo_id"])
    check("kv_identity_pure_attention", not bad_ident, f"{len(bad_ident)} bad e.g. {bad_ident[:5]}")
    check("geometry_matches_raw_config", not bad_match, f"{len(bad_match)} bad e.g. {bad_match[:5]}")
    checks["geometry_unverifiable"] = "; ".join(unverifiable)

    gated = [l for l in GATED_TXT.read_text().splitlines() if l.strip()] if GATED_TXT.exists() else []
    checks["gated_skipped_count"] = str(len(gated))
    checks["n_rows"] = str(len(rows))
    checks["rows_per_year"] = json.dumps(dict(sorted(years.items())), sort_keys=True)
    checks["rows_per_org"] = json.dumps(
        dict(sorted(Counter(r["org"] for r in rows).items())), sort_keys=True)
    checks["attention_type_counts"] = json.dumps(
        dict(sorted(Counter(r["attention_type"] for r in rows).items(), key=lambda x: -x[1])),
        sort_keys=False)
    checks["n_distinct_orgs"] = str(len({r["org"] for r in rows}))
    checks["n_dedup_groups"] = str(len({r["dedup_group"] for r in rows}))
    checks["n_params_from_safetensors"] = str(sum(1 for r in rows if r["params_source"] == "safetensors"))
    checks["n_params_from_config"] = str(sum(1 for r in rows if r["params_source"] == "config"))
    checks["n_params_missing"] = str(sum(1 for r in rows if r["params_source"] == "none"))
    kv_err = json.loads(KV_ERRORS.read_text()) if KV_ERRORS.exists() else []
    checks["n_kv_errors"] = str(len(kv_err))
    checks["kv_errors_sample"] = json.dumps([e["repo_id"] + " :: " + e["error"] for e in kv_err[:10]])
    return checks, ok


def main() -> int:
    rows = build()
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in COLUMNS})
    checks, ok = verify(rows)
    REPORT.write_text(json.dumps(checks, indent=1, sort_keys=True))
    for k, v in checks.items():
        print(f"{k}: {v}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
