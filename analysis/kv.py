"""
kv-census analytic KV-cache accounting.

Pure functions over a raw Hugging Face ``config.json`` **dict**.  No network
access, no transformers import: the caller supplies the dict (the census
builder reads it from ``data/raw/configs/``).  All results are the
*theoretical minimum* cache/state bytes for ONE sequence; implementation
divergences (HF allocating a full sliding window, MLA cached decompressed,
mamba conv state padded to ``d_conv``) are measured by the validation
experiment (RQ4), not modelled here.

Conventions and detection rules
===============================

Bytes-per-token
    A full-attention layer caches, per token, a K and a V tensor of
    ``n_kv_heads x head_dim`` elements each:
    ``bytes_per_token = 2 * n_kv_heads * head_dim * bytes_per_element``.

Head-dim rule
    Explicit ``head_dim`` in the config wins; otherwise
    ``hidden_size // num_attention_heads``.  (Gemma family: ``head_dim=256``
    while ``hidden_size // heads`` is smaller - the explicit value is used.)
    For MLA configs the reported head dim is
    ``qk_nope_head_dim + qk_rope_head_dim`` (the decompressed query/key width).

KV heads rule
    ``num_key_value_heads`` if present and non-null, else
    ``num_attention_heads`` (i.e. MHA).  ``mha`` = kv heads equal query heads,
    ``mqa`` = 1 kv head, ``gqa`` = anything between.  Legacy ChatGLM configs
    use ``multi_query_attention`` / ``multi_query_group_num`` / ``kv_channels``.

Sliding-window layers
    A windowed layer holds ``min(T, W)`` tokens at context ``T``.  How a layer
    is classified as windowed, per family (family = ``model_type``, falling
    back to ``architectures[0]``):

    * **mistral / llama / qwen2 / qwen3 (+_moe) / mixtral** style: windowed
      iff ``use_sliding_window`` is truthy AND ``sliding_window`` is a positive
      int.  (Qwen2.5 ships ``sliding_window=131072`` with the flag False and
      Mistral-7B-v0.3 ships ``sliding_window=null`` - both are full attention.)
    * **gemma2 / gemma3_text** (gemma3 multimodal is unwrapped to its
      ``text_config`` first):

      1. explicit ``layer_types`` list wins (``sliding_attention`` -> window,
         ``full_attention`` -> full).  Note: several Gemma-2 re-uploads
         serialize the 1:1 alternation; the original release semantics are
         5 local : 1 global via ``sliding_window_pattern``.
      2. else ``sliding_window_pattern`` = p (int, default 6): layer i is
         global iff ``(i + 1) % p == 0`` (last layer of every p-block),
         matching transformers' Gemma3TextConfig.
      3. else if only ``sliding_window`` is set: alternate every 2 layers
         (transformers>=5 Gemma2TextConfig default), global iff (i+1)%2==0.
      4. no ``sliding_window`` -> all full.
    * **gpt_oss**: explicit ``layer_types`` (real configs alternate
      ``sliding_attention``/``full_attention`` 1:1 with W=``sliding_window``
      =128; RoPE scaling is irrelevant to KV), same %2 default if absent.
    * **starcoder2**: windowed iff ``sliding_window`` is a positive int
      (Starcoder2 always applies its window; there is no flag).
    * **recurrent_gemma**: attention blocks use the local window
      ``attention_window_size``.
    * **phi3 / phi**: the ``sliding_window`` key in Phi-3 configs is vestigial
      (HF Phi-3 attention is full causal attention) -> always full.

MLA (latent) layers
    ``deepseek_v2`` / ``deepseek_v3`` / ``deepseek_v32`` / ``deepseek_v4`` /
    ``kimi_k2``(=DeepseekV3 arch) / full-attention layers of ``kimi_linear``:
    the theoretical latent cache is
    ``(kv_lora_rank + qk_rope_head_dim)`` elements per token (compressed K +
    RoPE'd K; V is decompressed on the fly and is not cached).  IMPORTANT:
    ``first_k_dense_replace`` in these configs switches the *FFN* between
    dense and MoE (HF: ``layer.mlp``), it does NOT change attention; ALL
    attention layers of these architectures are MLA (verified against
    transformers ``deepseek_v2/modeling_deepseek_v2.py`` and DeepSeek-V2/V3
    checkpoints).

Recurrent layers (constant state, independent of T)
    * **mamba1** (``jamba``, pure ``mamba``): d_inner = expand*hidden;
      ssm state ``d_inner * d_state`` elements; conv state
      ``d_inner * (d_conv - 1)`` elements (the conv runs over d_inner
      channels only).
    * **mamba2** (``granitemoehybrid``/``granite_moe_hybrid``,
      ``nemotron_h``, ``falcon_h1``, pure ``mamba2``, ``zamba2``): d_inner =
      n_heads*head_dim (= expand*hidden); ssm state
      ``d_inner * d_state`` elements; conv state
      ``(d_inner + 2*n_groups*d_state) * (d_conv - 1)`` elements - the
      causal conv runs over d_inner + 2*n_groups*d_state channels (x, B, C
      with B/C grouped; verified against transformers mamba2/nemotron_h/
      falcon_h1 model files).
    * **rglru** (``recurrent_gemma``, Griffin): rnn state ``lru_width``
      elements + conv state ``hidden * (conv1d_width - 1)``.
    * **gdn** gated-delta-net / KDA (``qwen3_next``, ``kimi_linear``
      linear layers, ``lfm2`` gdn blocks): rnn state
      ``num_v_heads * head_v_dim * head_k_dim`` elements (+ conv state;
      qwen3_next conv = ``(2*key_dim + value_dim) * (kernel - 1)``,
      kimi_linear conv = ``3 * qkv_dim * (kernel - 1)`` where
      qkv_dim = linear_num_heads*linear_head_dim, lfm2 conv =
      ``conv_dim * (conv_L_cache - 1)``).
    * **lightning** (``minimax``/``minimax_text_01`` linear layers):
      matrix state ``num_attention_heads * head_dim**2`` elements.
    * **rwkv4** (``rwkv``): ``5 * hidden`` elements per layer
      (3 attention-state vectors + 2 channel-mix vectors; HF cache is 5
      tensors of hidden).  rwkv5/rwkv6: ``n_head*head_size**2 + hidden``
      per layer if ``head_size`` is in the config, else KVConfigError.

Hybrid layer patterns
    * ``layers_block_type`` (jamba): values ``attention`` (GQA full) /
      ``mamba`` / ``mlp`` / ``expert`` (FFN-only, no cache).
    * ``layer_types`` (granitemoehybrid, lfm2, kimi_linear, qwen3_next,
      minimax, gpt_oss): values as above; ``linear_attention`` = recurrent.
      Jamba also supports derived patterns via ``attn_layer_period`` /
      ``attn_layer_offset`` (attention iff ``i % period == offset``).
    * ``hybrid_override_pattern`` (nemotron_h): single chars, mapping taken
      verbatim from transformers ``configuration_nemotron_h.py``:
      ``M``=linear_attention (mamba), ``E``=moe (FFN-only mixer, no cache),
      ``*``=full_attention, ``-``=mlp (FFN-only mixer, no cache).  Granite-
      style patterns (if a granite config ships one) use
      ``A``=attention, ``M``=mamba, ``D``/``E``=FFN-only.
    * ``falcon_h1``: EVERY decoder layer contains BOTH a GQA attention and a
      mamba2 mixer (verified in transformers falcon_h1 modeling), so each
      layer contributes one full-KV entry AND one recurrent entry.
    * ``lfm2``: ``full_attn_idxs`` lists the (0-based) global-attention
      layers; every other layer is a gdn block (rnn + conv state).
    * ``kimi_linear``: ``linear_attn_config.full_attn_layers`` /
      ``kda_layers`` are 1-BASED; default is full attention at 1-based
      positions 4, 8, 12, ... Full layers are MLA latents.
    * ``qwen3_next``: full attention iff ``(i+1) % full_attention_interval``
      == 0 (interval default 4); the rest are gdn layers.
    * ``minimax``/``minimax_text_01``: ``attn_type_list[i] == 1`` -> full
      softmax attention, ``0`` -> lightning (linear) attention.
    * nemotron_h / falcon_h1 / jamba attention layers are FULL attention
      (``sliding_window=null`` in their configs).

Cross-layer KV sharing
    Detected ONLY from explicit config keys (no known model in the census
    carries them): ``num_kv_shared_layers`` / ``kv_sharing`` /
    ``shared_kv_layers`` with an int value ``k >= 2`` means each group of k
    consecutive attention layers shares one KV cache; the shared cache is
    emitted (and counted) ONCE per group.  A ``model_type``/architecture name
    containing ``yoco`` is treated as one shared cache for the whole stack.

Unknown architectures
    Anything not in the registry, or a registered family with missing /
    inconsistent geometry fields, raises :class:`KVConfigError` - never a
    silent guess.  The census builder records those rows separately.

Multimodal nesting
    :func:`text_scope` picks the language backbone from ``text_config`` ->
    ``llm_config`` -> ``language_config`` -> ``text_model`` (first one that
    looks like a text backbone: has ``num_hidden_layers`` or
    ``num_attention_heads``), else the top-level dict.  (Gemma-3 and LLaVA
    use ``text_config``; InternVL uses ``language_config``; Qwen2-VL is flat.)

KV-parity convention (kv_par_context)
    Returns the smallest integer T >= 1 with ``kv_bytes_at(T) >=
    total_params * weight_bytes_per_param``.  For models whose cache is
    bounded (windows and/or recurrent only): if the cache saturates below
    parity, return the saturation context (max window size W; 0 -> no
    windows), documented as a marker; a constant cache (pure recurrent)
    that never reaches parity returns ``math.inf``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "KVConfigError",
    "LayerKV",
    "text_scope",
    "parse_layers",
    "kv_bytes_per_token",
    "kv_bytes_at",
    "fixed_state_bytes",
    "describe",
    "kv_par_context",
]


class KVConfigError(ValueError):
    """Raised when a config cannot be parsed into a KV accounting."""


@dataclass(frozen=True)
class LayerKV:
    kind: str                  # "full" | "window" | "latent" | "recurrent"
    subtype: str               # "mha"|"gqa"|"mqa"|"swa"|"mla"|"mamba2"|"mamba"|"linear"|"gdn"|"deltanet"|"rwkv"|...
    n_kv_heads: int            # 0 when not applicable (latent/recurrent)
    head_dim: int              # 0 when not applicable
    bytes_per_element: int = 2  # 2 (BF16) by default
    window: int | None = None  # max cached tokens for "window" layers, None otherwise
    latent_elems: int | None = None  # per-token latent elements for "latent" (MLA)
    fixed_state_bytes: float = 0.0  # constant per-layer state bytes (recurrent layers), else 0.0


# --------------------------------------------------------------------------- #
# small config helpers
# --------------------------------------------------------------------------- #

_TEXT_SCOPES = ("text_config", "llm_config", "language_config", "text_model")


def text_scope(config: dict) -> dict:
    """Return the language-backbone sub-dict of a (possibly multimodal) config."""
    if not isinstance(config, dict):
        raise KVConfigError("config must be a dict")
    for key in _TEXT_SCOPES:
        sub = config.get(key)
        if isinstance(sub, dict) and ("num_hidden_layers" in sub or "num_attention_heads" in sub):
            return sub
    return config


def _model_type(cfg: dict) -> str:
    mt = cfg.get("model_type")
    if isinstance(mt, str) and mt:
        return mt
    archs = cfg.get("architectures")
    if isinstance(archs, list) and archs and isinstance(archs[0], str):
        return archs[0].removesuffix("ForCausalLM").removesuffix("ForConditionalGeneration").lower()
    # original state-spaces release format: {"d_model":..., "n_layer":..., "ssm_cfg": {...}}
    if "d_model" in cfg and "n_layer" in cfg:
        ssm = cfg.get("ssm_cfg")
        ssm = ssm if isinstance(ssm, dict) else {}
        if ssm.get("layer") == "Mamba2" or "headdim" in ssm or "n_groups" in ssm:
            return "mamba2"
        return "mamba"
    raise KVConfigError("config has no model_type and no architectures entry")


def _req_int(cfg: dict, key: str, *, default=None, min_val=None) -> int:
    v = cfg.get(key, default)
    if v is None:
        raise KVConfigError(f"missing required config field '{key}' (model_type={cfg.get('model_type')})")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
        raise KVConfigError(f"config field '{key}' must be an integer, got {v!r}")
    v = int(v)
    if min_val is not None and v < min_val:
        raise KVConfigError(f"config field '{key}' must be >= {min_val}, got {v}")
    return v


def _opt_int(cfg: dict, key: str, default=None):
    v = cfg.get(key, default)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
        raise KVConfigError(f"config field '{key}' must be an integer, got {v!r}")
    return int(v)


def _num_layers(cfg: dict) -> int:
    n = _opt_int(cfg, "num_hidden_layers")
    if n is None:
        n = _opt_int(cfg, "n_layer")
    if n is None:
        raise KVConfigError("missing num_hidden_layers")
    return n


def _attn_geometry(cfg: dict) -> tuple[int, int, int]:
    """(num_attention_heads, num_kv_heads, head_dim) with the documented rules."""
    hidden = _req_int(cfg, "hidden_size", min_val=1)
    heads = _req_int(cfg, "num_attention_heads", min_val=1)
    kv = cfg.get("num_key_value_heads", heads)
    if kv is None:
        kv = heads
    if isinstance(kv, bool) or not isinstance(kv, (int, float)) or int(kv) != kv or int(kv) < 1:
        raise KVConfigError(f"bad num_key_value_heads: {kv!r}")
    kv = int(kv)
    hd = cfg.get("head_dim")
    if hd is None:
        if hidden % heads != 0:
            raise KVConfigError(
                f"hidden_size {hidden} not divisible by num_attention_heads {heads} "
                "and no explicit head_dim"
            )
        hd = hidden // heads
    elif isinstance(hd, bool) or not isinstance(hd, (int, float)) or int(hd) != hd or int(hd) < 1:
        raise KVConfigError(f"bad head_dim: {hd!r}")
    else:
        hd = int(hd)
    return heads, kv, hd


def _full_or_window(heads: int, kv: int, hd: int, window: int | None, bpe: int) -> LayerKV:
    if window:
        return LayerKV("window", "swa", kv, hd, bpe, window=int(window))
    sub = "mha" if kv == heads else ("mqa" if kv == 1 else "gqa")
    return LayerKV("full", sub, kv, hd, bpe)


# --------------------------------------------------------------------------- #
# layer-plan builders: each returns list[LayerKV]
# --------------------------------------------------------------------------- #

def _flag_window(cfg: dict) -> int | None:
    """mistral/qwen-style: window only when use_sliding_window is on."""
    if cfg.get("use_sliding_window") and isinstance(cfg.get("sliding_window"), (int, float)):
        w = int(cfg["sliding_window"])
        return w if w > 0 else None
    return None


def _gemma_window_plan(cfg: dict) -> list[str]:
    """'full' / 'window' plan per layer for gemma2 / gemma3_text."""
    n = _num_layers(cfg)
    w = cfg.get("sliding_window")
    w = int(w) if isinstance(w, (int, float)) and int(w) > 0 else None
    types = cfg.get("layer_types")
    if isinstance(types, list) and types:
        plan = []
        for t in types:
            t = str(t)
            if t in ("full_attention", "full", "global"):
                plan.append("full")
            elif t in ("sliding_attention", "sliding", "local"):
                plan.append("window" if w else "full")
            else:
                raise KVConfigError(f"unknown gemma layer_types entry {t!r}")
        if len(plan) != n:
            raise KVConfigError(f"layer_types length {len(plan)} != num_hidden_layers {n}")
        return plan
    if w is None:
        return ["full"] * n
    p = _opt_int(cfg, "sliding_window_pattern", 6)
    return ["full" if (i + 1) % p == 0 else "window" for i in range(n)]


def _plan_full(cfg: dict, bpe: int, *, window: int | None = None, plan: list[str] | None = None,
               layer_indices: list[int] | None = None) -> list[LayerKV]:
    """Standard GQA/MHA/MQA attention layers, optionally windowed / partial."""
    heads, kv, hd = _attn_geometry(cfg)
    n = _num_layers(cfg)
    idxs = layer_indices if layer_indices is not None else list(range(n))
    if plan is None:
        return [_full_or_window(heads, kv, hd, window, bpe) for _ in idxs]
    return [_full_or_window(heads, kv, hd, window if plan[i] == "window" else None, bpe) for i in idxs]


def _latent_layer(kv_lora: int, rope: int, bpe: int) -> LayerKV:
    return LayerKV("latent", "mla", 0, 0, bpe, latent_elems=kv_lora + rope)


def _mla(cfg: dict, bpe: int) -> list[LayerKV]:
    kv_lora = _req_int(cfg, "kv_lora_rank", min_val=1)
    rope = _req_int(cfg, "qk_rope_head_dim", min_val=1)
    n = _num_layers(cfg)
    return [_latent_layer(kv_lora, rope, bpe)] * n


# ---- mamba state helpers ---------------------------------------------------- #

def _mamba1_state_elems(cfg: dict) -> int:
    hidden = _req_int(cfg, "hidden_size", min_val=1)
    expand = _opt_int(cfg, "mamba_expand", _opt_int(cfg, "expand", 2))
    d_state = _opt_int(cfg, "mamba_d_state", _opt_int(cfg, "d_state", 16))
    d_conv = _opt_int(cfg, "mamba_d_conv", _opt_int(cfg, "d_conv", 4))
    d_inner = _opt_int(cfg, "mamba_d_inner") or expand * hidden
    if d_state < 1 or d_conv < 1:
        raise KVConfigError("bad mamba d_state/d_conv")
    ssm = d_inner * d_state
    conv = d_inner * (d_conv - 1)
    return ssm + conv


def _mamba2_state_elems(cfg: dict) -> int:
    """d_inner * d_state + (d_inner + 2*n_groups*d_state) * (d_conv-1) elements."""
    d_state = _opt_int(cfg, "ssm_state_size", _opt_int(cfg, "mamba_d_state", _opt_int(cfg, "d_state")))
    if d_state is None:
        d_state = 128
    n_groups = _opt_int(cfg, "n_groups", _opt_int(cfg, "mamba_n_groups", 1))
    d_conv = _opt_int(cfg, "conv_kernel", _opt_int(cfg, "d_conv", _opt_int(cfg, "mamba_d_conv", 4)))
    # d_inner: explicit conv_dimension / intermediate, else n_heads*head_dim, else expand*hidden
    d_inner = _opt_int(cfg, "conv_dimension")
    if d_inner is None:
        n_heads = _opt_int(cfg, "mamba_num_heads", _opt_int(cfg, "mamba_n_heads"))
        head_dim = _opt_int(cfg, "mamba_head_dim", _opt_int(cfg, "mamba_d_head"))
        if n_heads is not None and head_dim is not None:
            d_inner = n_heads * head_dim
        else:
            hidden = _req_int(cfg, "hidden_size", min_val=1)
            expand = _opt_int(cfg, "expand", _opt_int(cfg, "mamba_expand", 2))
            d_inner = expand * hidden
    if min(d_state, n_groups, d_conv, d_inner) < 1:
        raise KVConfigError(f"bad mamba2 geometry fields (d_inner={d_inner}, d_state={d_state}, n_groups={n_groups}, d_conv={d_conv})")
    ssm = d_inner * d_state
    conv_dim = _opt_int(cfg, "conv_dim") or (d_inner + 2 * n_groups * d_state)
    conv = conv_dim * (d_conv - 1)
    return ssm + conv


def _recurrent(cfg: dict, subtype: str, elems: int, bpe: int) -> LayerKV:
    if elems < 1:
        raise KVConfigError(f"non-positive recurrent state size for {subtype}")
    return LayerKV("recurrent", subtype, 0, 0, bpe, fixed_state_bytes=float(elems * bpe))


# ---- per-family builders ---------------------------------------------------- #

def _b_standard(cfg: dict, bpe: int) -> list[LayerKV]:
    """Plain GQA/MHA/MQA families; window only when the flag says so."""
    return _plan_full(cfg, bpe, window=_flag_window(cfg))


def _b_starcoder2(cfg: dict, bpe: int) -> list[LayerKV]:
    w = cfg.get("sliding_window")
    w = int(w) if isinstance(w, (int, float)) and int(w) > 0 else None
    return _plan_full(cfg, bpe, window=w)


def _b_phi3(cfg: dict, bpe: int) -> list[LayerKV]:
    return _plan_full(cfg, bpe, window=None)  # phi3 sliding_window is vestigial


def _b_gemma2(cfg: dict, bpe: int) -> list[LayerKV]:
    heads, kv, hd = _attn_geometry(cfg)
    w = cfg.get("sliding_window")
    w = int(w) if isinstance(w, (int, float)) and int(w) > 0 else None
    plan = _gemma_window_plan(cfg)
    return [
        _full_or_window(heads, kv, hd, w if p == "window" else None, bpe)
        for p in plan
    ]


def _b_gpt_oss(cfg: dict, bpe: int) -> list[LayerKV]:
    heads, kv, hd = _attn_geometry(cfg)
    n = _num_layers(cfg)
    w = cfg.get("sliding_window")
    w = int(w) if isinstance(w, (int, float)) and int(w) > 0 else None
    types = cfg.get("layer_types")
    if isinstance(types, list) and types:
        plan = ["full" if str(t) == "full_attention" else "window" for t in types]
        if len(plan) != n:
            raise KVConfigError(f"layer_types length {len(plan)} != num_hidden_layers {n}")
    else:
        plan = ["full" if (i + 1) % 2 == 0 else "window" for i in range(n)]
    return [_full_or_window(heads, kv, hd, w if p == "window" else None, bpe) for p in plan]


def _b_kimi_linear(cfg: dict, bpe: int) -> list[LayerKV]:
    n = _num_layers(cfg)
    lac = cfg.get("linear_attn_config") or {}
    full1: set[int] = set()
    if isinstance(lac, dict) and lac.get("full_attn_layers") and lac.get("kda_layers"):
        full1 = {int(x) for x in lac["full_attn_layers"]}  # 1-based layer numbers
        kda1 = {int(x) for x in lac["kda_layers"]}
        if full1 | kda1 != set(range(1, n + 1)):
            raise KVConfigError("kimi_linear full_attn_layers/kda_layers do not partition 1..num_hidden_layers")
    else:
        full1 = {(i + 1) for i in range(n) if (i + 1) % 4 == 0}
    kv_lora = _req_int(cfg, "kv_lora_rank", min_val=1)
    rope = _req_int(cfg, "qk_rope_head_dim", min_val=1)
    num_heads = _opt_int(lac if isinstance(lac, dict) else {}, "num_heads", _opt_int(cfg, "linear_num_heads"))
    head_dim = _opt_int(lac if isinstance(lac, dict) else {}, "head_dim", _opt_int(cfg, "linear_head_dim"))
    kernel = _opt_int(lac if isinstance(lac, dict) else {}, "short_conv_kernel_size",
                      _opt_int(cfg, "linear_conv_kernel_dim", 4))
    if num_heads is None or head_dim is None:
        raise KVConfigError("kimi_linear config lacks linear num_heads/head_dim")
    qkv_dim = num_heads * head_dim
    conv_elems = 3 * qkv_dim * (kernel - 1)
    rnn_elems = num_heads * head_dim * head_dim  # (heads, head_k, head_v) with head_k == head_v
    out = []
    for i in range(1, n + 1):
        if i in full1:
            out.append(_latent_layer(kv_lora, rope, bpe))
        else:
            out.append(_recurrent(cfg, "gdn", rnn_elems + conv_elems, bpe))
    return out


_NEMOTRON_PATTERN = {"M": "linear_attention", "E": "moe", "*": "full_attention", "-": "mlp"}
_GRANITE_PATTERN = {"A": "full_attention", "M": "linear_attention", "D": "mlp", "E": "moe",
                    "*": "full_attention", "-": "mlp"}


def _pattern_to_types(pattern: str, mapping: dict[str, str]) -> list[str]:
    if not pattern or not isinstance(pattern, str):
        raise KVConfigError("empty hybrid_override_pattern")
    out = []
    for ch in pattern:
        if ch not in mapping:
            raise KVConfigError(f"unknown hybrid_override_pattern letter {ch!r}")
        out.append(mapping[ch])
    return out


def _layer_types_list(cfg: dict, n: int, *, aliases: dict[str, str]) -> list[str]:
    """Normalize layers_block_type / layer_types / hybrid_override_pattern."""
    types = cfg.get("layers_block_type")
    if not isinstance(types, list) or not types:
        types = cfg.get("layer_types")
    if isinstance(types, list) and types:
        norm = []
        for t in types:
            t = aliases.get(str(t), str(t))
            norm.append(t)
        if len(norm) != n:
            raise KVConfigError(f"layer list length {len(norm)} != num_hidden_layers {n}")
        return norm
    return None


_ATTN_ALIASES = {
    "attention": "full_attention", "full": "full_attention",
    "sliding_attention": "sliding_attention", "swa": "sliding_attention",
    "linear_attention": "linear_attention", "mamba": "linear_attention",
    "recurrent": "linear_attention", "gdn": "linear_attention", "kda": "linear_attention",
    "mlp": "mlp", "dense": "mlp", "moe": "moe", "expert": "moe",
    "sparse_moe_block": "moe", "conv": "linear_attention",
}


def _b_jamba(cfg: dict, bpe: int) -> list[LayerKV]:
    n = _num_layers(cfg)
    types = _layer_types_list(cfg, n, aliases=_ATTN_ALIASES)
    if types is None:
        period = _opt_int(cfg, "attn_layer_period", 8)
        offset = _opt_int(cfg, "attn_layer_offset", 4)
        types = ["full_attention" if i % period == offset else "linear_attention" for i in range(n)]
    heads, kv, hd = _attn_geometry(cfg)
    elems = _mamba1_state_elems(cfg)
    out = []
    for t in types:
        if t == "full_attention":
            out.append(_full_or_window(heads, kv, hd, None, bpe))  # jamba attn is full
        elif t == "linear_attention":
            out.append(_recurrent(cfg, "mamba", elems, bpe))
        else:  # mlp / moe: FFN-only layer, no cache
            continue
    return out


def _mamba2_hybrid(cfg: dict, bpe: int, *, pattern_mapping: dict | None,
                   per_layer_attention_and_mamba: bool = False) -> list[LayerKV]:
    n = _num_layers(cfg)
    types = _layer_types_list(cfg, n, aliases=_ATTN_ALIASES)
    if types is None and pattern_mapping is not None and cfg.get("hybrid_override_pattern"):
        types = _pattern_to_types(cfg["hybrid_override_pattern"], pattern_mapping)
        if len(types) != n:
            raise KVConfigError(
                f"hybrid_override_pattern length {len(types)} != num_hidden_layers {n}"
            )
    heads, kv, hd = _attn_geometry(cfg)
    elems = _mamba2_state_elems(cfg)
    out = []
    if per_layer_attention_and_mamba:
        # falcon_h1: every layer owns BOTH an attention KV and a mamba state
        if types is not None:
            raise KVConfigError("falcon_h1 config unexpectedly carries a layer pattern")
        for _ in range(n):
            out.append(_full_or_window(heads, kv, hd, None, bpe))
            out.append(_recurrent(cfg, "mamba2", elems, bpe))
        return out
    if types is None:
        raise KVConfigError("hybrid config has neither layers_block_type/layer_types nor hybrid_override_pattern")
    for t in types:
        if t == "full_attention":
            out.append(_full_or_window(heads, kv, hd, None, bpe))
        elif t == "linear_attention":
            out.append(_recurrent(cfg, "mamba2", elems, bpe))
        elif t in ("mlp", "moe"):
            continue
        else:
            raise KVConfigError(f"unknown layer type {t!r}")
    return out


def _b_recurrent_gemma(cfg: dict, bpe: int) -> list[LayerKV]:
    n = _num_layers(cfg)
    types = cfg.get("block_types")
    if isinstance(types, (list, tuple)) and types:
        bt = [str(t) for t in types]
    else:
        bt = ["recurrent", "recurrent", "attention"]
    hidden = _req_int(cfg, "hidden_size", min_val=1)
    lru = _opt_int(cfg, "lru_width", hidden)
    conv_w = _opt_int(cfg, "conv1d_width", 4)
    win = _opt_int(cfg, "attention_window_size")
    heads, kv, hd = _attn_geometry(cfg)
    rnn_elems = lru + hidden * (conv_w - 1)
    out = []
    for i in range(n):
        t = bt[i % len(bt)]
        if t in ("recurrent", "linear_attention", "mamba"):
            out.append(_recurrent(cfg, "rglru", rnn_elems, bpe))
        elif t in ("attention", "full_attention"):
            if not win:
                raise KVConfigError("recurrent_gemma attention block needs attention_window_size")
            out.append(_full_or_window(heads, kv, hd, win, bpe))
        else:
            raise KVConfigError(f"unknown recurrent_gemma block_types entry {t!r}")
    return out


def _b_lfm2(cfg: dict, bpe: int) -> list[LayerKV]:
    n = _num_layers(cfg)
    idxs = cfg.get("full_attn_idxs")
    if isinstance(idxs, list) and idxs:
        full = {int(x) for x in idxs}
        if not full <= set(range(n)):
            raise KVConfigError("full_attn_idxs out of range")
    else:
        full = {i for i in range(n) if (i + 1) % 4 == 0}
    hidden = _req_int(cfg, "hidden_size", min_val=1)
    num_heads = _opt_int(cfg, "num_heads", _opt_int(cfg, "gdn_num_heads"))
    if num_heads is None:
        raise KVConfigError("lfm2 config lacks GDN num_heads")
    if hidden % num_heads != 0:
        raise KVConfigError(f"lfm2 hidden_size {hidden} not divisible by gdn num_heads {num_heads}")
    head = hidden // num_heads
    conv_dim = _opt_int(cfg, "conv_dim", hidden)
    conv_k = _opt_int(cfg, "conv_L_cache", 3)
    rnn_elems = num_heads * head * head
    conv_elems = conv_dim * (conv_k - 1)
    heads, kv, hd = _attn_geometry(cfg)
    out = []
    for i in range(n):
        if i in full:
            out.append(_full_or_window(heads, kv, hd, None, bpe))
        else:
            out.append(_recurrent(cfg, "gdn", rnn_elems + conv_elems, bpe))
    return out


def _b_qwen3_next(cfg: dict, bpe: int) -> list[LayerKV]:
    n = _num_layers(cfg)
    types = _layer_types_list(cfg, n, aliases=_ATTN_ALIASES)
    if types is None:
        interval = _opt_int(cfg, "full_attention_interval", 4)
        types = ["linear_attention" if (i + 1) % interval else "full_attention" for i in range(n)]
    heads, kv, hd = _attn_geometry(cfg)
    nk = _opt_int(cfg, "linear_num_key_heads")
    nv = _opt_int(cfg, "linear_num_value_heads")
    hk = _opt_int(cfg, "linear_key_head_dim")
    hv = _opt_int(cfg, "linear_value_head_dim")
    kernel = _opt_int(cfg, "linear_conv_kernel_dim", 4)
    if None in (nk, nv, hk, hv):
        raise KVConfigError("qwen3_next config lacks linear_* head geometry")
    rnn_elems = nv * hv * hk
    conv_elems = (2 * nk * hk + nv * hv) * (kernel - 1)
    out = []
    for t in types:
        if t == "full_attention":
            out.append(_full_or_window(heads, kv, hd, None, bpe))
        else:
            out.append(_recurrent(cfg, "gdn", rnn_elems + conv_elems, bpe))
    return out


def _b_minimax(cfg: dict, bpe: int) -> list[LayerKV]:
    n = _num_layers(cfg)
    heads, kv, hd = _attn_geometry(cfg)
    attn_list = cfg.get("attn_type_list")
    out = []
    if isinstance(attn_list, list) and len(attn_list) == n:
        for t in attn_list:
            if int(t) == 1:
                out.append(_full_or_window(heads, kv, hd, None, bpe))
            else:
                out.append(_recurrent(cfg, "lightning", heads * hd * hd, bpe))
        return out
    types = _layer_types_list(cfg, n, aliases=_ATTN_ALIASES)
    if types is None:
        raise KVConfigError("minimax config has neither attn_type_list nor layer_types")
    for t in types:
        if t == "full_attention":
            out.append(_full_or_window(heads, kv, hd, None, bpe))
        elif t == "linear_attention":
            out.append(_recurrent(cfg, "lightning", heads * hd * hd, bpe))
        else:
            raise KVConfigError(f"unknown minimax layer type {t!r}")
    return out


def _b_mamba_pure(cfg: dict, bpe: int, *, mamba2: bool) -> list[LayerKV]:
    # original state-spaces configs use d_model/n_layer; allow ssm_cfg overrides
    if "hidden_size" not in cfg and "d_model" in cfg:
        cfg = {**cfg, "hidden_size": cfg["d_model"]}
    ssm = cfg.get("ssm_cfg")
    if isinstance(ssm, dict):
        cfg = {**cfg, **{k: v for k, v in ssm.items() if k in
                         ("d_state", "expand", "d_conv", "n_groups", "headdim")}}
    n = _num_layers(cfg)
    if mamba2:
        # original Mamba2 defaults: d_state=128, n_groups=8, headdim=64, expand=2
        d_state = _opt_int(cfg, "d_state", 128)
        n_groups = _opt_int(cfg, "n_groups", 8)
        head = _opt_int(cfg, "headdim", 64)
        expand = _opt_int(cfg, "expand", 2)
        d_conv = _opt_int(cfg, "d_conv", 4)
        hidden = _req_int(cfg, "hidden_size", min_val=1)
        d_inner = expand * hidden
        if d_inner % head != 0:
            raise KVConfigError("mamba2 d_inner not divisible by headdim")
        elems = d_inner * d_state + (d_inner + 2 * n_groups * d_state) * (d_conv - 1)
        sub = "mamba2"
    else:
        elems = _mamba1_state_elems(cfg)
        sub = "mamba"
    return [_recurrent(cfg, sub, elems, bpe) for _ in range(n)]


def _b_rwkv(cfg: dict, bpe: int) -> list[LayerKV]:
    mt = _model_type(cfg)
    n = _num_layers(cfg)
    if mt == "rwkv":  # RWKV-4
        hidden = _req_int(cfg, "hidden_size", min_val=1)
        elems = 5 * hidden
    else:  # rwkv5 / rwkv6 legacy
        hidden = _req_int(cfg, "hidden_size", min_val=1)
        head_size = _opt_int(cfg, "head_size")
        if head_size is None:
            raise KVConfigError(f"{mt} state needs 'head_size' in the config")
        if hidden % head_size != 0:
            raise KVConfigError("hidden_size not divisible by head_size")
        elems = (hidden // head_size) * head_size * head_size + hidden
    return [_recurrent(cfg, "rwkv", elems, bpe) for _ in range(n)]


def _b_chatglm(cfg: dict, bpe: int) -> list[LayerKV]:
    hidden = _req_int(cfg, "hidden_size", min_val=1)
    heads = _req_int(cfg, "num_attention_heads", min_val=1)
    if cfg.get("multi_query_attention"):
        kv = _req_int(cfg, "multi_query_group_num", min_val=1)
        hd = _req_int(cfg, "kv_channels", min_val=1)
    else:
        kv = heads
        hd = _opt_int(cfg, "kv_channels") or hidden // heads
    n = _num_layers(cfg)
    return [LayerKV("full", "mha" if kv == heads else ("mqa" if kv == 1 else "gqa"), kv, hd, bpe)
            for _ in range(n)]


# ---- registry --------------------------------------------------------------- #

_STANDARD_FLAGGED = {
    # plain GQA/MHA/MQA, window only behind use_sliding_window
    "llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe",
    "internlm2", "internlm3", "baichuan", "baichuan2", "olmo", "olmo2", "olmoe",
    "stablelm", "phi", "gemma", "glm", "glm4", "glm4_moe", "nemotron",
    "exaone", "exaone4", "exaone_moe", "cohere", "command-r", "smollm", "smollm2",
    "smollm3", "arcee", "granite", "granitemoe", "phimoe", "hunyuan",
    "hunyuan_v1_dense", "hunyuan_v1_moe", "gpt_neox", "mllama", "chatglm2", "chatglm3",
}
_MLA = {"deepseek_v2", "deepseek_v3", "deepseek_v32", "deepseek_v4", "kimi_k2", "kimi_k25"}

_REGISTRY: dict[str, object] = {
    **{mt: _b_standard for mt in _STANDARD_FLAGGED},
    "starcoder2": _b_starcoder2,
    "phi3": _b_phi3,
    "gemma2": _b_gemma2,
    "gemma3_text": _b_gemma2,
    "gemma3": _b_gemma2,          # text_scope already unwrapped text_config
    "gpt_oss": _b_gpt_oss,
    **{mt: _mla for mt in _MLA},
    "kimi_linear": _b_kimi_linear,
    "jamba": _b_jamba,
    "granitemoehybrid": lambda c, b: _mamba2_hybrid(c, b, pattern_mapping=_GRANITE_PATTERN),
    "granite_moe_hybrid": lambda c, b: _mamba2_hybrid(c, b, pattern_mapping=_GRANITE_PATTERN),
    "zamba2": lambda c, b: _mamba2_hybrid(c, b, pattern_mapping=_GRANITE_PATTERN),
    "nemotron_h": lambda c, b: _mamba2_hybrid(c, b, pattern_mapping=_NEMOTRON_PATTERN),
    "falcon_h1": lambda c, b: _mamba2_hybrid(c, b, pattern_mapping=_GRANITE_PATTERN,
                                             per_layer_attention_and_mamba=True),
    "recurrent_gemma": _b_recurrent_gemma,
    "lfm2": _b_lfm2,
    "lfm2_moe": _b_lfm2,
    "qwen3_next": _b_qwen3_next,
    "minimax": _b_minimax,
    "minimax_text_01": _b_minimax,
    "mamba": lambda c, b: _b_mamba_pure(c, b, mamba2=False),
    "mamba2": lambda c, b: _b_mamba_pure(c, b, mamba2=True),
    "rwkv": _b_rwkv,
    "rwkv5": _b_rwkv,
    "rwkv6": _b_rwkv,
    "chatglm": _b_chatglm,
    # You-Only-Cache-Once style: standard geometry + one cross-layer shared cache
    # (detected by name; see _detect_sharing).
    "yoco": _b_standard,
}

_SHARING_KEYS = ("num_kv_shared_layers", "kv_sharing", "shared_kv_layers")


def _detect_sharing(cfg: dict) -> int | None:
    """k >= 2 => each group of k consecutive attention layers shares one cache."""
    for key in _SHARING_KEYS:
        v = cfg.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and int(v) >= 2:
            return int(v)
    name = f"{cfg.get('model_type', '')}{''.join(cfg.get('architectures') or [])}".lower()
    if "yoco" in name:
        return 10**9  # whole stack shares one cache
    return None


def _apply_sharing(layers: list[LayerKV], k: int) -> list[LayerKV]:
    out: list[LayerKV] = []
    attn_i = 0
    for layer in layers:
        if layer.kind in ("full", "window", "latent"):
            if attn_i % k == 0:
                out.append(layer)
            attn_i += 1
        else:
            out.append(layer)
    return out


def parse_layers(config: dict, *, bytes_per_element: int = 2) -> list[LayerKV]:
    """One LayerKV per cache-owning unit (shared caches emitted once)."""
    bpe = bytes_per_element
    if not isinstance(bpe, int) or bpe < 1:
        raise KVConfigError(f"bytes_per_element must be a positive int, got {bpe!r}")
    cfg = text_scope(config)
    mt = _model_type(cfg)
    builder = _REGISTRY.get(mt)
    if builder is None:
        raise KVConfigError(f"unknown/unregistered architecture (model_type={mt!r})")
    layers = builder(cfg, bpe)
    if not layers:
        raise KVConfigError(f"empty layer plan for model_type={mt!r}")
    k = _detect_sharing(cfg)
    if k:
        layers = _apply_sharing(layers, k)
    return layers


# --------------------------------------------------------------------------- #
# byte accounting
# --------------------------------------------------------------------------- #

def _layer_bytes_at(layer: LayerKV, T: int) -> float:
    if layer.kind == "full":
        return 2.0 * layer.n_kv_heads * layer.head_dim * layer.bytes_per_element * T
    if layer.kind == "window":
        return 2.0 * layer.n_kv_heads * layer.head_dim * layer.bytes_per_element * min(T, layer.window or 0)
    if layer.kind == "latent":
        return float((layer.latent_elems or 0) * layer.bytes_per_element * T)
    return float(layer.fixed_state_bytes)


def _layer_marginal(layer: LayerKV) -> float:
    if layer.kind in ("full", "latent"):
        return _layer_bytes_at(layer, 1)
    return 0.0


def kv_bytes_per_token(config: dict, *, bytes_per_element: int = 2) -> float:
    """Asymptotic marginal bytes per extra token (T -> inf): full + latent only."""
    return float(sum(_layer_marginal(l) for l in parse_layers(config, bytes_per_element=bytes_per_element)))


def kv_bytes_at(config: dict, T: int, *, bytes_per_element: int = 2) -> float:
    """Total cache+state bytes for ONE sequence at context length T (theoretical minimum)."""
    if not isinstance(T, (int, float)) or isinstance(T, bool) or T < 0:
        raise KVConfigError(f"T must be a non-negative number, got {T!r}")
    T = int(T)
    return float(sum(_layer_bytes_at(l, T) for l in parse_layers(config, bytes_per_element=bytes_per_element)))


def fixed_state_bytes(config: dict, *, bytes_per_element: int = 2) -> float:
    """Sum of constant per-layer state bytes (recurrent layers only)."""
    return float(sum(l.fixed_state_bytes for l in parse_layers(config, bytes_per_element=bytes_per_element)
                     if l.kind == "recurrent"))


def kv_par_context(config: dict, *, total_params: float, weight_bytes_per_param: float = 2.0,
                   bytes_per_element: int = 2) -> float:
    """Smallest integer T >= 1 with kv_bytes_at(T) >= total_params*weight_bytes_per_param.

    Convention for bounded caches (windows and/or recurrent state only, no
    unbounded layers): if the cache saturates below parity, the saturation
    context (the largest window W) is returned as a marker; a windowless,
    constant cache (pure recurrent) that cannot reach parity returns
    math.inf.  If the bounded cache does reach parity (small model), the
    normal smallest-T rule applies.
    """
    if total_params <= 0:
        raise KVConfigError("total_params must be positive")
    layers = parse_layers(config, bytes_per_element=bytes_per_element)
    target = float(total_params) * float(weight_bytes_per_param)

    def at(t: int) -> float:
        return float(sum(_layer_bytes_at(l, t) for l in layers))

    slope = sum(_layer_marginal(l) for l in layers)
    windows = [l.window for l in layers if l.kind == "window"]
    sat_T = max(windows) if windows else 0
    if slope > 0:
        hi = max(sat_T, math.ceil(target / slope) + 2, 1)
    else:
        hi = sat_T
    if at(max(hi, 1)) < target:
        # cache can never reach parity
        return float(sat_T) if windows else math.inf
    lo, best = 1, None
    hi = max(hi, 1)
    while lo <= hi:
        mid = (lo + hi) // 2
        if at(mid) >= target:
            best, hi = mid, mid - 1
        else:
            lo = mid + 1
    return float(best if best is not None else math.inf)


# --------------------------------------------------------------------------- #
# describe
# --------------------------------------------------------------------------- #

def describe(config: dict) -> dict:
    cfg = text_scope(config)
    layers = parse_layers(config)
    arch = cfg.get("architectures")
    arch = arch[0] if isinstance(arch, list) and arch else cfg.get("model_type")
    attn_layers = [l for l in layers if l.kind in ("full", "window", "latent")]
    first_attn = attn_layers[0] if attn_layers else None
    heads = _opt_int(cfg, "num_attention_heads")
    kv = first_attn.n_kv_heads if first_attn is not None and first_attn.n_kv_heads else None
    hidden = _opt_int(cfg, "hidden_size") or _opt_int(cfg, "d_model")
    if kv is None and heads is not None:
        kv = _opt_int(cfg, "num_key_value_heads", heads)

    n_full = sum(1 for l in layers if l.kind == "full")
    n_window = sum(1 for l in layers if l.kind == "window")
    n_latent = sum(1 for l in layers if l.kind == "latent")
    n_rec = sum(1 for l in layers if l.kind == "recurrent")
    windows = sorted({l.window for l in layers if l.kind == "window"})
    window_size = windows[0] if len(windows) == 1 else (max(windows) if windows else None)

    # attention_type label
    rec_subtypes = {l.subtype for l in layers if l.kind == "recurrent"}
    if n_rec and attn_layers:
        if "mamba" in rec_subtypes or "mamba2" in rec_subtypes:
            atype = "hybrid-mamba"
        elif "gdn" in rec_subtypes:
            atype = "hybrid-gdn"
        elif "rglru" in rec_subtypes:
            atype = "hybrid-lru"
        else:
            atype = f"hybrid-{sorted(rec_subtypes)[0]}"
    elif n_rec and not attn_layers:
        atype = sorted(rec_subtypes)[0]
    elif n_latent and n_window:
        atype = "mla+swa"
    elif n_latent:
        atype = "mla"
    elif n_window and n_full:
        atype = "local+global"
    elif n_window:
        sub = attn_layers[0].subtype if attn_layers else "gqa"
        # windowed layers are always built as subtype "swa"; recover gqa-ness
        hd, h = (attn_layers[0].n_kv_heads, attn_layers[0].head_dim) if attn_layers else (None, None)
        if h is not None and heads:
            gsub = "mha" if h == heads else ("mqa" if h == 1 else "gqa")
        else:
            gsub = "gqa"
        atype = f"{gsub}+swa"
    else:
        a = attn_layers[0]
        atype = a.subtype

    head_dim = first_attn.head_dim if first_attn is not None else None
    if not head_dim and n_latent and "qk_nope_head_dim" in cfg and cfg.get("qk_nope_head_dim"):
        head_dim = _opt_int(cfg, "qk_nope_head_dim") + _opt_int(cfg, "qk_rope_head_dim")
    gqa_ratio = (heads / kv) if (heads and kv) else None
    return {
        "arch": arch,
        "num_layers": _num_layers(cfg),
        "num_attention_heads": heads,
        "num_kv_heads": kv,
        "head_dim": head_dim,
        "hidden_size": hidden,
        "attention_type": atype,
        "gqa_ratio": gqa_ratio,
        "window_size": window_size,
        "n_full_layers": n_full,
        "n_window_layers": n_window,
        "n_latent_layers": n_latent,
        "n_recurrent_layers": n_rec,
        "kv_lora_rank": _opt_int(cfg, "kv_lora_rank") if n_latent else None,
        "qk_rope_head_dim": _opt_int(cfg, "qk_rope_head_dim") if n_latent else None,
        "kv_shared_cross_layer": _detect_sharing(cfg) is not None,
        "kv_bytes_per_token_bf16": kv_bytes_per_token(config),
        "kv_bytes_at_32k": kv_bytes_at(config, 32_768),
        "kv_bytes_at_128k": kv_bytes_at(config, 131_072),
        "recurrent_state_bytes": fixed_state_bytes(config),
    }
