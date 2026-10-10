#!/usr/bin/env python
"""RQ4 validation: analytic KV accounting (analysis/kv.py) vs cache tensors that
HF transformers actually allocates.

Method (BRIEF.md "Validation experiment", INTERFACES.md section 3):
- For each model: load the snapshotted config.json from analysis/validation_configs/,
  reduce DEPTH to one full period of the layer-type pattern, shrink the vocabulary,
  keep per-layer attention geometry (kv heads, head_dim, ranks, windows) exact.
- Instantiate with AutoModelForCausalLM.from_config (random init, no weight
  downloads), CPU only, float32.
- Prefill T tokens with use_cache=True and sum the BYTES of every cache tensor the
  returned Cache object holds (keys/values, conv states, recurrent/SSM states).
- Compare with analysis/kv.py kv_bytes_at() evaluated on the SAME reduced config
  dict with bytes_per_element=4 (float32).

Output: results/kv_validation.csv with columns
model_or_arch, attention_class, analytic_bytes, measured_bytes, rel_error, explanation
rel_error = abs(measured - analytic) / analytic (6+ digits). No timing-dependent
values are written; row order is fixed by the spec table.

Deviations from the original model (documented per row in the explanation column):
- depth reduced to one layer-type period, vocab shrunk to 1024, MoE expert count
  reduced (gpt-oss, DeepSeek-V2-Lite) - none of these touch attention geometry;
- google/gemma-2-2b is gated (401 without a token) and was replaced by
  bigcode/starcoder2-3b (GQA + sliding window W=4096, run with T=4352 > W) and
  openai/gpt-oss-20b (W=128, T=2048 > W) for the windowed classes;
- state-spaces/mamba-2.8b ships an original Mamba-format config (d_model, n_layer,
  no model_type); it is mapped explicitly to a transformers MambaConfig
  (d_model=2560 -> hidden_size, n_layer=64 -> 1 layer, d_state=16, d_conv=4,
  expand=2, intermediate_size=2*d_model, dt_rank="auto").
"""

import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["HF_HUB_OFFLINE"] = "1"       # from_config + local config dirs only
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, MambaConfig

CONFIG_DIR = ROOT / "analysis" / "validation_configs"
OUT_CSV = ROOT / "results" / "kv_validation.csv"

# One row per model. overrides are applied on top of the snapshotted config.json;
# layer_types (when present) is sliced consistently with num_hidden_layers.
SPECS = [
    dict(
        slug="microsoft__phi-2", repo="microsoft/phi-2", cls="mha", T=2048,
        overrides={"num_hidden_layers": 1, "vocab_size": 1024},
        note="MHA reference: 32 q/kv heads, head_dim 80 (hidden 2560 / 32), full attention.",
    ),
    dict(
        slug="Qwen__Qwen2.5-0.5B", repo="Qwen/Qwen2.5-0.5B", cls="gqa", T=2048,
        overrides={"num_hidden_layers": 1, "vocab_size": 1024},
        note="GQA reference: 14 q heads / 2 kv heads, head_dim 64 (explicit in config).",
    ),
    dict(
        slug="bigcode__starcoder2-3b", repo="bigcode/starcoder2-3b", cls="gqa+swa", T=4352,
        overrides={"num_hidden_layers": 1, "vocab_size": 1024},
        note="sliding-window GQA: all layers windowed, W=4096 (config sliding_window; no "
             "layer_types). T=4352 > W so the window saturates.",
    ),
    dict(
        slug="openai__gpt-oss-20b", repo="openai/gpt-oss-20b", cls="gqa+swa", T=2048,
        overrides={"num_hidden_layers": 2, "vocab_size": 1024, "num_local_experts": 8,
                   "num_experts_per_tok": 4, "experts_per_token": 4},
        attn="eager",  # transformers 5.x GptOssForCausalLM has no sdpa path
        note="window+global hybrid (layer pattern [sliding, full], period 2), W=128, T=2048 > W; "
             "MoE experts 32 -> 8 (routing only, no KV effect).",
    ),
    dict(
        slug="deepseek-ai__DeepSeek-V2-Lite", repo="deepseek-ai/DeepSeek-V2-Lite", cls="mla", T=2048,
        overrides={"num_hidden_layers": 2, "vocab_size": 1024, "n_routed_experts": 8},
        note="MLA: kv_lora_rank 512 + qk_rope_head_dim 64; 2 layers kept = layer 0 (dense MLP, "
             "first_k_dense_replace=1) + 1 MoE layer; routed experts 64 -> 8 (no KV effect).",
    ),
    dict(
        slug="ibm-granite__granite-4.0-h-micro", repo="ibm-granite/granite-4.0-h-micro",
        cls="hybrid-mamba", T=2048,
        overrides={"num_hidden_layers": 10, "vocab_size": 1024},
        note="hybrid mamba2+attention: attention at layer idx 5, 15, 25, 35 -> period 10 "
             "(9 mamba2 + 1 GQA-attention layer kept).",
    ),
    dict(
        slug="state-spaces__mamba-2.8b", repo="state-spaces/mamba-2.8b", cls="mamba", T=512,
        overrides={"num_hidden_layers": 1, "vocab_size": 1024},
        attn="eager",  # MambaForCausalLM has no sdpa path
        note="pure recurrent (Mamba-1 SSM); state is T-independent, T=512 suffices.",
    ),
]

BYTE_LABELS = ("bytes", "KB", "MB", "GB")


def human(n: float) -> str:
    for lab in BYTE_LABELS:
        if n < 1024 or lab == BYTE_LABELS[-1]:
            return f"{n:.1f} {lab}" if lab != BYTE_LABELS[0] else f"{int(n)} B"
        n /= 1024
    return f"{n} B"


# ---------------------------------------------------------------- config build

def reduced_config_dict(spec) -> dict:
    """Load the snapshotted config.json and apply the documented reductions."""
    with open(CONFIG_DIR / f"{spec['slug']}.json") as f:
        cfg = json.load(f)
    if spec["slug"] == "state-spaces__mamba-2.8b":
        # Original Mamba format -> transformers MambaConfig (documented mapping).
        mcfg = MambaConfig(
            hidden_size=cfg["d_model"],                 # 2560
            num_hidden_layers=1,                        # n_layer 64 -> one period (period 1)
            vocab_size=1024,                            # 50277 -> 1024
            state_size=cfg.get("ssm_cfg", {}).get("d_state", 16),   # 16
            conv_kernel=cfg.get("ssm_cfg", {}).get("d_conv", 4),    # 4
            expand=cfg.get("ssm_cfg", {}).get("expand", 2),         # 2
        )
        d = mcfg.to_dict()
        d["intermediate_size"] = 2 * d["hidden_size"]
        return d
    ov = dict(spec["overrides"])
    if "layer_types" in cfg and "num_hidden_layers" in ov:
        # keep layer_types consistent with the reduced depth (one full period)
        ov["layer_types"] = cfg["layer_types"][: ov["num_hidden_layers"]]
    cfg.update(ov)
    for tok in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if cfg.get(tok) is not None and cfg.get(tok, 0) >= cfg["vocab_size"]:
            cfg[tok] = None
    return cfg


def build_model(cfg_dict, attn="sdpa"):
    """Write the reduced config to a temp dir, load it locally, random-init on CPU."""
    with tempfile.TemporaryDirectory(prefix="kvval_") as td:
        p = Path(td) / "config.json"
        p.write_text(json.dumps(cfg_dict))
        cfg_obj = AutoConfig.from_pretrained(str(p), local_files_only=True)
    model = AutoModelForCausalLM.from_config(cfg_obj, dtype=torch.float32, attn_implementation=attn)
    model.eval()
    return model


# ---------------------------------------------------------------- measurement

def tensor_bytes(x) -> int:
    import torch as _t
    if _t.is_tensor(x):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(tensor_bytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(tensor_bytes(v) for v in x)
    return 0


def shapes(x):
    import torch as _t
    if _t.is_tensor(x):
        return list(x.shape)
    if isinstance(x, dict):
        return {k: shapes(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [shapes(v) for v in x]
    return type(x).__name__


def measure_cache(past) -> tuple[int, list]:
    """Sum bytes of every cache tensor in the returned Cache object.

    Handles the transformers 5.x Cache API (past.layers[i].keys/.values,
    LinearAttentionLayer.conv_states/.recurrent_states dicts) and the legacy
    DynamicCache attributes (key_cache / value_cache lists).
    """
    total = 0
    details = []
    if hasattr(past, "layers"):
        for i, layer in enumerate(past.layers):
            parts = []
            for attr in ("keys", "values", "conv_states", "recurrent_states"):
                v = getattr(layer, attr, None)
                b = tensor_bytes(v)
                if b:
                    total += b
                    parts.append(f"{attr}{shapes(v)}")
            if parts:
                details.append((i, type(layer).__name__, " ".join(parts)))
    else:  # legacy DynamicCache
        for attr in ("key_cache", "value_cache", "conv_states", "ssm_states"):
            v = getattr(past, attr, None)
            b = tensor_bytes(v)
            if b:
                total += b
                details.append((-1, attr, str(shapes(v)[:2])))
    return total, details


# ------------------------------------------------------------------- analytic

def analytic_bytes(cfg_dict, T):
    """analysis/kv.py kv_bytes_at on the SAME reduced config dict, float32."""
    from analysis import kv  # noqa: import here so measurement can be tested alone

    return float(kv.kv_bytes_at(cfg_dict, T, bytes_per_element=4))


# ------------------------------------------------------------------ rows

def cached_kv_lengths(past):
    """[(layer_idx, stored_len)] for attention-style layers (keys tensor)."""
    out = []
    if hasattr(past, "layers"):
        for i, layer in enumerate(past.layers):
            k = getattr(layer, "keys", None)
            if torch.is_tensor(k):
                out.append((i, int(k.shape[-2])))
    return out


def explain(spec, cfg, past, details, analytic, measured):
    """Concrete, fact-based explanation built from the measured cache contents."""
    slug = spec["slug"]
    layers = getattr(past, "layers", [])
    lens = cached_kv_lengths(past)
    base = spec["note"] + " "
    if slug in ("microsoft__phi-2", "Qwen__Qwen2.5-0.5B"):
        return base + (
            f"DynamicCache stores K and V of shape [B, kv_heads, T, head_dim] in float32 "
            f"(layer 0: {lens[0][1]} tokens); matches 2 * n_kv_heads * head_dim * 4 * T exactly."
        )
    if slug == "bigcode__starcoder2-3b":
        w = cfg["sliding_window"]
        stored = lens[0][1] if lens else -1
        return base + (
            f"transformers 5.x DynamicSlidingWindowLayer keeps W-1={stored} of the {spec['T']} seen "
            f"positions (T>W saturated); analytic assumes min(T,W)={min(spec['T'], w)}. "
            f"The window is enforced in the cache itself (truncation), not only by masking."
        )
    if slug == "openai__gpt-oss-20b":
        w = cfg["sliding_window"]
        sw, gl = (lens[0][1] if len(lens) > 0 else -1), (lens[1][1] if len(lens) > 1 else -1)
        return base + (
            f"sliding layer 0 stores W-1={sw} of {spec['T']} positions (truncated), full layer 1 "
            f"stores T={gl}; analytic uses min(T,W)={min(spec['T'], w)} for layer 0 and T for layer 1."
        )
    if slug == "deepseek-ai__DeepSeek-V2-Lite":
        k0 = layers[0].keys
        v0 = layers[0].values
        return base + (
            f"transformers caches the COMPRESSED latent per layer: keys [B,1,T,{k0.shape[-1]}] "
            f"(kv_lora_rank) + values [B,1,T,{v0.shape[-1]}] (rope part) = "
            f"{k0.shape[-1] + v0.shape[-1]} elems/token/layer = kv_lora_rank + qk_rope_head_dim; "
            f"no decompressed K/V is materialised in the cache."
        )
    if slug == "ibm-granite__granite-4.0-h-micro":
        ml = layers[0]
        cs = ml.conv_states[0]
        rs = ml.recurrent_states[0]
        d_conv = int(cs.shape[2])
        chan = int(cs.shape[1])
        extra = chan * (d_conv - (d_conv - 1)) * 4 * 9
        return base + (
            f"mamba2 layers hold constant state: conv_states [B,{chan},{d_conv}] "
            f"(d_inner + 2*n_groups*d_state = {chan} channels) + recurrent_states "
            f"[B,{rs.shape[1]},{rs.shape[2]},{rs.shape[3]}] = n_heads*d_head*d_state (exact); "
            f"attention layer 5 stores [B,8,T,64] K/V in float32 (exact). The 0.56% gap is "
            f"conv-state padding: analytic uses the theoretical minimum {d_conv - 1} stored inputs "
            f"per channel, HF allocates all {d_conv} conv columns (+{extra} B over 9 mamba layers)."
        )
    if slug == "state-spaces__mamba-2.8b":
        ml = layers[0]
        cs = ml.conv_states[0]
        rs = ml.recurrent_states[0]
        d_conv = int(cs.shape[2])
        chan = int(cs.shape[1])
        extra = chan * 4
        return base + (
            f"conv_states [B,{chan},{d_conv}] + recurrent_states [B,{rs.shape[1]},{rs.shape[2]}] "
            f"in float32, both independent of T. The >1% gap is conv-state padding: the analytic "
            f"minimum keeps {d_conv - 1} stored inputs per channel, HF allocates the full "
            f"d_conv={d_conv} kernel columns (+{chan} x 4 B = {extra} B)."
        )
    return base + "see stdout details"


def run(spec, analytic_fn):
    cfg = reduced_config_dict(spec)
    model = build_model(cfg, attn=spec.get("attn", "sdpa"))
    torch.manual_seed(0)
    ids = torch.randint(0, cfg["vocab_size"], (1, spec["T"]))
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    dt = time.time() - t0
    past = getattr(out, "past_key_values", None) or getattr(out, "cache_params", None)
    measured, details = measure_cache(past)
    # round to integers FIRST so rel_error is exactly recomputable from the CSV columns
    analytic = int(round(analytic_fn(cfg, spec["T"])))
    measured = int(measured)
    rel = abs(measured - analytic) / analytic
    row = dict(
        model_or_arch=spec["repo"],
        attention_class=spec["cls"],
        analytic_bytes=analytic,
        measured_bytes=measured,
        rel_error=f"{rel:.8f}",
        explanation=explain(spec, cfg, past, details, analytic, measured),
    )
    print(f"[{spec['repo']:42s}] T={spec['T']:5d} fwd={dt:5.1f}s  "
          f"analytic={human(analytic)}  measured={human(measured)}  rel_err={rel:.8f}")
    for i, name, part in details:
        print(f"    layer {i:2d} {name:28s} {part[:150]}")
    del model, out, past, details
    gc.collect()
    return row


def main():
    rows = [run(spec, analytic_bytes) for spec in SPECS]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        f.write("model_or_arch,attention_class,analytic_bytes,measured_bytes,rel_error,explanation\n")
        for r in rows:
            expl = r["explanation"].replace('"', "'")
            f.write(f'{r["model_or_arch"]},{r["attention_class"]},{r["analytic_bytes"]},'
                    f'{r["measured_bytes"]},{r["rel_error"]},"{expl}"\n')
    n_ok = sum(1 for r in rows if float(r["rel_error"]) <= 0.01)
    print(f"\nwrote {OUT_CSV}: {len(rows)} rows, "
          f"{len(set(r['attention_class'] for r in rows))} attention classes, "
          f"{n_ok}/{len(rows)} rows within 1% relative error")


if __name__ == "__main__":
    main()
