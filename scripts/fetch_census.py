#!/usr/bin/env python3
"""Stage A: enumerate open-weight decoder-only LLM checkpoints from original
publisher orgs on the Hugging Face Hub and snapshot everything needed offline.

Census window : Hub createdAt in [2023-01-01, today] (UTC date).

Outputs
  data/raw/hub_api/list__<org>__p<N>.json     raw listing pages (provenance)
  data/raw/hub_api/list__<org>__merged.json   merged per-org listing
  data/raw/hub_api/info__<org>__<name>.json   raw per-repo model-info response
  data/raw/hub_api/cfgstat__<org>__<name>.json HTTP status of the config fetch
  data/raw/configs/<org>__<name>.json         raw config.json of included repos
  data/raw/census_manifest.json               per-repo decision record
  data/raw/census_summary.json                machine-readable summary
  data/gated_skipped.txt                      gated repos (config 401/403)

Policy (AGENTS.md / BRIEF.md / task brief)
  * no HF token; plain User-Agent; <= ~3.1 req/s to huggingface.co only
  * every network response cached on disk; complete-cache re-runs are offline
  * quantized re-uploads / converted artifacts excluded (tags + name tokens)
  * gated repos whose config.json download fails with 401/403 are skipped and
    counted (selection-bias limitation)

Original publisher orgs: the task-mandated list plus additions justified here
  meta-llama  : BRIEF.md example publisher (Llama 2/3/3.x). Repos are
                license-gated; config.json is not anonymously downloadable, so
                they are expected to land in gated_skipped.txt. Enumerated so
                the skip count is complete and honest.
  EleutherAI  : original publisher of the Pythia suite (Hub uploads 2023-02+).
  RWKV        : original publisher of RWKV-4/5/6/7 decoder LLMs; recurrent
                state, needed for the census' state-type coverage.
  mosaicml    : original publisher of MPT-7B/30B (2023; later Databricks).
  databricks  : original publisher of DBRX (2024).
  Snowflake   : original publisher of Arctic MoE (2024).
  LGAI-EXAONE : original publisher of EXAONE 3.x/4.x (2024-2025).
  NOT included: mistral-community (not a publisher) and any aggregator or
  fine-tune account (unsloth, TheBloke, bartowski, mlx-community, ...).
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
HUB_API = ROOT / "data" / "raw" / "hub_api"
CONFIGS = ROOT / "data" / "raw" / "configs"
MANIFEST = ROOT / "data" / "raw" / "census_manifest.json"
SUMMARY = ROOT / "data" / "raw" / "census_summary.json"
GATED_TXT = ROOT / "data" / "gated_skipped.txt"
for d in (HUB_API, CONFIGS):
    d.mkdir(parents=True, exist_ok=True)

MIN_DATE = "2023-01-01"
MAX_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")
USER_AGENT = "kv-census/0.1 (census of open-weight LLMs on HF; anonymous, no token)"
MIN_INTERVAL = 0.5  # seconds between requests (2 req/s, well under the 4 req/s cap)

ORGS = [
    # task-mandated original publishers
    "mistralai", "Qwen", "google", "google-deepmind", "microsoft",
    "deepseek-ai", "01-ai", "tiiuae", "allenai", "nvidia", "ibm-granite",
    "CohereForAI", "CohereLabs", "zai-org", "THUDM", "moonshotai", "MiniMaxAI",
    "openai", "internlm", "baichuan-inc", "stabilityai", "HuggingFaceTB",
    "apple", "LiquidAI", "ServiceNow-AI", "ai21labs", "openbmb", "bigcode",
    "core42", "upstage", "stepfun-ai", "tencent", "inclusionAI",
    # additions justified in the module docstring
    "meta-llama", "EleutherAI", "RWKV", "mosaicml", "databricks",
    "Snowflake", "LGAI-EXAONE",
]

# tags marking derivative artifacts (quantized / converted / exported formats)
DERIVATIVE_TAGS = {
    "onnx", "diffusers", "sentence-transformers", "coreml", "openvino",
    "gguf", "ggml", "mlx", "compressed-tensors", "optimum", "TensorRT",
    "trtllm", "awq", "gptq", "bitsandbytes", "autogptq", "autoawq", "hqq",
    "eetq", "aqlm", "marlin", "exl2", "exl3", "ctranslate2",
}

# pipeline tags that cannot belong to a decoder-only text LLM checkpoint
NON_LLM_PIPELINES = {
    "feature-extraction", "sentence-similarity", "text-classification",
    "token-classification", "fill-mask", "translation", "summarization",
    "question-answering", "zero-shot-classification",
    "table-question-answering", "text2text-generation", "text-to-image",
    "image-to-image", "unconditional-image-generation", "depth-estimation",
    "object-detection", "image-segmentation", "image-to-text",
    "zero-shot-image-classification", "mask-generation", "text-to-speech",
    "automatic-speech-recognition", "audio-classification",
    "audio-text-to-text", "text-to-audio", "audio-to-audio",
    "voice-activity-detection", "time-series-forecasting",
    "time-series-imputation", "time-series-classification",
    "video-classification", "reinforcement-learning", "robotics", "other",
    "graph-ml", "keypoint-detection", "pose-estimation", "text-to-video",
    "video-to-video",
}

# tags marking non-LLM artifacts regardless of pipeline_tag (image generation /
# editing, TTS codecs); kept separate from DERIVATIVE_TAGS (format re-uploads)
NON_LLM_TAGS = {
    "text-to-image", "image-generation", "image-editing", "prompt-rewriting",
    "judge-model", "tts", "codec", "voice-clone",
}

# name tokens (case-insensitive) that mark quantized/converted re-uploads
QUANT_NAME_TOKENS = {
    "awq", "gptq", "gguf", "ggml", "int2", "int4", "int8", "2bit", "3bit",
    "4bit", "6bit", "8bit", "fp4", "fp6", "fp8", "nf4", "nvfp4", "q2", "q3",
    "q4", "q4k", "q5", "q6", "q8", "bnb", "mlx", "onnx", "openvino", "ct2",
    "ctranslate2", "exl2", "exl3", "trt", "tensorrt", "trtllm", "hqq",
    "aqlm", "eetq", "marlin", "quantized", "quant", "bnb4bit", "compressed",
    "mxfp4", "mxfp8", "w4a8", "w8a8", "w4a16", "w8a16", "int4blocks",
}
_NAME_TOKEN_SPLIT = re.compile(r"[-_.]+")

# model_type denylist: clearly not a decoder text LLM (encoder, encoder-decoder,
# vision/audio/diffusion/time-series); anything unknown goes to "unclassified"
NON_TEXT_MODEL_TYPES = {
    "bert", "roberta", "xlm-roberta", "nomic_bert", "deberta", "distilbert",
    "albert", "electra", "camembert", "mpnet", "xmod", "modernbert",
    "clip", "clip_vision_model", "siglip", "siglip_vision_model", "vit",
    "vit_mae", "dinov2", "dinov3", "dino", "beit", "swin", "convnext",
    "poolformer", "hiera", "levit", "efficientnet", "regnet", "resnet",
    "sam", "sam_hq", "detr", "maskformer", "mask2former", "yolos", "owlvit",
    "owlv2", "rt_detr", "grounding-dino", "depth_anything", "zoedepth",
    "imagebind", "videomae", "videomae2", "timesformer", "perceiver",
    "data2vec-vision", "data2vec-audio", "data2vec-text",
    "whisper", "wav2vec2", "wav2vec2-bert", "hubert", "sew", "sew-d",
    "unispeech", "unispeech-sat", "speech_to_text", "encodec", "hifigan",
    "musicgen", "audioldm2", "vits", "bark", "clvp", "parakeet", "canary",
    "canary_1b", "titan", "pop2piano",
    "stable-diffusion", "sd3", "flux", "sdxl", "controlnet", "unet",
    "autoencoderkl", "vae", "latent-consistency",
    "t5", "mt5", "umt5", "byt5", "longt5", "flan-t5", "switch_transformers",
    "nllb-moe", "nllb", "m2m_100", "bart", "mbart", "marian", "led",
    "pegasus", "prophetnet", "speecht5", "vision-encoder-decoder",
    "moment", "timesfm", "moirai", "chronos", "informer", "autoformer",
    "time_series_transformer", "timer",
    "git",  # image-captioning VLM, not a general text LLM
}

# multimodal containers whose config nests a decoder text backbone
MM_TEXT_MODEL_TYPES = {
    "llava", "llava_next", "llava_onevision", "qwen2_vl", "qwen2_5_vl",
    "qwen3_vl", "qwen_vl", "qwen2_5_omni", "qwen2_audio", "paligemma",
    "paligemma2", "gemma3", "gemma3n", "granitevision", "idefics",
    "idefics2", "idefics3", "pixtral", "aya_vision", "glm4v", "glm4v_moe",
    "glm4_1v", "minicpmv", "internvl_chat", "kimi_vl", "deepseek_vl",
    "deepseek_vl2", "mllama", "chameleon", "internvl", "phi3_v",
    "smolvlm", "imp", "ovis", "phi4_multimodal", "multi_modality",
    # 2025-2026 families, verified to nest a text_config language backbone
    "qwen3_5", "qwen3_5_moe", "qwen4_exp", "qwen3_omni_moe", "qwen3_vl_moe",
    "exaone4_5", "lfm2_vl", "minimax_vl_01", "minimax_m3_vl", "cohere2_vision",
}
MM_ARCH_EXACT = {"InternVLChatModel", "MiniCPMV"}

TEXT_ARCH_EXACT = {"ChatGLMModel"}

# HTTP session, anonymous by construction; rate limited
_last_req = [0.0]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


def http_get(url: str, params=None, *, tries=8):
    for attempt in range(tries):
        wait = MIN_INTERVAL - (time.time() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()
        try:
            r = SESSION.get(url, params=params, timeout=40)
        except requests.RequestException as e:
            _log(f"  net error {url[:90]}: {e!r}; attempt {attempt + 1}")
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            try:
                ra = float(ra) if ra is not None else 0.0
            except ValueError:
                ra = 0.0
            backoff = min(max(ra, 10.0 * (attempt + 1)), 120.0)
            _log(f"  429 at {url[:90]}; sleeping {backoff:.0f}s (attempt {attempt + 1})")
            time.sleep(backoff)
            continue
        if r.status_code >= 500:
            _log(f"  {r.status_code} at {url[:90]}; attempt {attempt + 1}")
            time.sleep(10 * (attempt + 1))
            continue
        return r
    raise RuntimeError(f"HTTP failed after {tries} tries: {url}")


def cached_json(path: Path, *, url, params=None):
    """Load JSON from cache; fetch+store on miss. Non-200 -> cached marker."""
    if path.exists():
        return json.loads(path.read_text())
    r = http_get(url, params=params)
    if r.status_code == 200:
        try:
            data = r.json()
        except json.JSONDecodeError:
            data = {"_http_status": r.status_code, "_error": "not_json"}
        else:
            data["_fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path.write_text(json.dumps(data, sort_keys=True))
        return data
    path.write_text(json.dumps({"_http_status": r.status_code}))
    return {"_http_status": r.status_code}


def fetch_org_listing(org: str) -> list[dict]:
    """All model repos of one org (cursor pagination over /api/models)."""
    merged_path = HUB_API / f"list__{org}__merged.json"
    if merged_path.exists():
        return json.loads(merged_path.read_text())["items"]

    items: list[dict] = []
    page, seen, url = 0, set(), "https://huggingface.co/api/models"
    params = {"author": org, "limit": 1000}
    while True:
        page_path = HUB_API / f"list__{org}__p{page}.json"
        if page_path.exists():
            data = json.loads(page_path.read_text())
        else:
            r = http_get(url, params=params)
            if r.status_code in (400, 404):
                _log(f"org {org}: listing status {r.status_code} -> empty")
                merged_path.write_text(json.dumps({"org": org, "items": []}))
                return []
            data = {"_list": r.json(), "_link_next": r.headers.get("Link", "")}
            page_path.write_text(json.dumps(data))
        page_items = data.get("_list") or []
        for it in page_items:
            if it["id"] not in seen:
                seen.add(it["id"])
                items.append(it)
        nxt = data.get("_link_next", "")
        if 'rel="next"' not in nxt or not page_items:
            break
        # keep the full next URL verbatim (the cursor is already percent-encoded)
        url = nxt[nxt.index("<") + 1 : nxt.index(">")]
        params = None
        page += 1
    merged_path.write_text(
        json.dumps({"org": org, "n_items": len(items), "items": items}, sort_keys=True)
    )
    _log(f"org {org}: {len(items)} repos listed")
    return items


def prefilter(item: dict) -> tuple[str, str]:
    """Cheap listing-stage filter. Returns (ok, reason)."""
    rid = item["id"]
    tags = set(item.get("tags") or [])
    name = rid.split("/", 1)[1] if "/" in rid else rid
    created = (item.get("createdAt") or "")[:10]
    if not created:
        return False, "no_createdAt"
    if created < MIN_DATE or created > MAX_DATE:
        return False, "outside_window"
    if tags & DERIVATIVE_TAGS:
        return False, "derivative_tag:" + ",".join(sorted(tags & DERIVATIVE_TAGS))
    if tags & NON_LLM_TAGS:
        return False, "non_llm_tag:" + ",".join(sorted(tags & NON_LLM_TAGS))
    tokens = {t.lower() for t in _NAME_TOKEN_SPLIT.split(name) if t}
    hit = tokens & QUANT_NAME_TOKENS
    # generic "<N>bit" tokens (1.58bit, 2bit, 4bit, ...) mark compressed variants;
    # native low-bit architectures (e.g. microsoft bitnet-b1.58-2B-4T) do NOT
    # match this pattern ("b1" and "58" are separate tokens)
    bitlike = {t for t in tokens if re.fullmatch(r"\d+(\.\d+)?bit", t)}
    if hit or bitlike:
        return False, "quant_name:" + ",".join(sorted(hit | bitlike))
    pipe = item.get("pipeline_tag")
    if pipe in NON_LLM_PIPELINES:
        return False, f"pipeline:{pipe}"
    return True, ""


def classify(info: dict) -> tuple[str, str]:
    """(ok, reason) from the per-repo info summary."""
    cfg = info.get("config") or {}
    model_type = cfg.get("model_type") or ""
    archs = cfg.get("architectures") or []
    arch = archs[0] if archs else ""
    if model_type in NON_TEXT_MODEL_TYPES:
        return False, f"non_text_model_type:{model_type}"
    is_text_arch = (
        arch.endswith(("ForCausalLM", "LMHeadModel", "TextModel"))
        or "rwkv" in arch.lower()
        or arch in TEXT_ARCH_EXACT
        or model_type in {"rwkv"}   # HF RWKV4/5 port: valid config, no archs
    )
    if is_text_arch:
        return True, "text"
    if model_type in MM_TEXT_MODEL_TYPES or arch in MM_ARCH_EXACT:
        return True, "multimodal_text"
    return False, f"unclassified:{model_type or 'none'}:{arch or 'none'}"


def main() -> int:
    manifest = []
    for org in ORGS:
        items = fetch_org_listing(org)
        cands = []
        for it in items:
            ok, reason = prefilter(it)
            if ok:
                cands.append(it)
            else:
                manifest.append(
                    {"repo_id": it["id"], "org": org, "decision": "excluded",
                     "reason": reason, "created_at": it.get("createdAt")}
                )
        _log(f"org {org}: {len(cands)} candidates after prefilter")

        for it in sorted(cands, key=lambda x: x["id"]):
            rid = it["id"]
            name = rid.split("/", 1)[1]
            info_path = HUB_API / f"info__{org}__{name}.json"
            info = cached_json(info_path, url=f"https://huggingface.co/api/models/{rid}")
            status = info.get("_http_status", 200)
            if status != 200 or "config" not in info:
                manifest.append(
                    {"repo_id": rid, "org": org, "decision": "excluded",
                     "reason": f"info_http_{status}", "created_at": it.get("createdAt")}
                )
                continue
            ok, reason = classify(info)
            entry = {
                "repo_id": rid,
                "org": org,
                "name": name,
                "created_at": info.get("createdAt") or it.get("createdAt"),
                "gated": info.get("gated"),
                "pipeline_tag": info.get("pipeline_tag"),
                "safetensors_total": (info.get("safetensors") or {}).get("total"),
                "safetensors_parameters": (info.get("safetensors") or {}).get("parameters"),
                "model_type": (info.get("config") or {}).get("model_type"),
                "architectures": (info.get("config") or {}).get("architectures"),
                "library_name": info.get("library_name"),
                "tags": info.get("tags"),
            }
            if not ok:
                entry.update({"decision": "excluded", "reason": reason})
                manifest.append(entry)
                continue

            # config.json snapshot (the geometry ground truth)
            cfg_path = CONFIGS / f"{org}__{name}.json"
            stat_path = HUB_API / f"cfgstat__{org}__{name}.json"
            http_status = None
            config_ok = False
            if cfg_path.exists():
                config_ok, http_status = True, 200
            elif stat_path.exists():
                cached_status = json.loads(stat_path.read_text())["status"]
                if cached_status in (401, 403):
                    http_status = cached_status
            if not config_ok and http_status is None:
                r = http_get(f"https://huggingface.co/{rid}/resolve/main/config.json")
                http_status = r.status_code
                if http_status == 404:  # rare non-main default branch
                    r = http_get(f"https://huggingface.co/{rid}/resolve/master/config.json")
                    http_status = r.status_code
                if http_status == 200:
                    try:
                        cfg = json.loads(r.content)
                        if isinstance(cfg, dict):
                            cfg_path.write_text(json.dumps(cfg, sort_keys=True))
                            config_ok = True
                        else:
                            http_status = "not_dict"
                    except json.JSONDecodeError:
                        http_status = "not_json"
                stat_path.write_text(json.dumps({"status": http_status}))
            if config_ok:
                entry.update({"decision": "included", "reason": reason,
                              "config_file": str(cfg_path.relative_to(ROOT))})
            elif http_status in (401, 403):
                entry.update({"decision": "gated_skipped",
                              "reason": f"config_http_{http_status}"})
            else:
                entry.update({"decision": "excluded",
                              "reason": f"config_http_{http_status}"})
            manifest.append(entry)

    manifest.sort(key=lambda e: e["repo_id"])
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True))

    gated = sorted({e["repo_id"] for e in manifest if e["decision"] == "gated_skipped"})
    GATED_TXT.write_text("\n".join(gated) + ("\n" if gated else ""))

    inc = [e for e in manifest if e["decision"] == "included"]
    per_org = {}
    for e in inc:
        per_org[e["org"]] = per_org.get(e["org"], 0) + 1
    reasons = {}
    for e in manifest:
        if e["decision"] != "included":
            key = e["reason"].split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    years = {}
    for e in inc:
        y = (e.get("created_at") or "")[:4]
        years[y] = years.get(y, 0) + 1
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": [MIN_DATE, MAX_DATE],
        "n_manifest": len(manifest),
        "n_included": len(inc),
        "n_gated_skipped": len(gated),
        "included_per_org": dict(sorted(per_org.items())),
        "included_per_year": dict(sorted(years.items())),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "unclassified": sorted(
            e["repo_id"] + " -> " + e["reason"]
            for e in manifest if e["reason"].startswith("unclassified")
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True))
    _log(f"DONE: included={len(inc)} gated_skipped={len(gated)}")
    _log("included_per_year=" + json.dumps(summary["included_per_year"], sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
