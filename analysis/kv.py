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
      3. else ``sliding_window_pattern`` int p (default 6 = the documented
         5:1 local:global design): layer i is global iff ``(i + 1) % p == 0``.
         (transformers>=5 serializes the 1:1 alternation explicitly instead,
         which case 1 already covers.)
      4. no ``sliding_window`` -> all full.
    The same plan drives gemma4_text, olmo3, exaone4_5_text ("LLLG" string
    patterns), step3p5, cohere2(_moe), bolmo, granite_switch and
    qwen3_scale_seq configs that carry ``layer_types`` / pattern fields.
    granite_swa/granitemoe_swa default: full iff ``i % 4 == 0``, else
    window W=``sliding_window`` (128).
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
    * **mamba1** (``jamba``, pure ``mamba``, ``falcon_mamba``): d_inner =
      expand*hidden;
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

Census wave-2 conventions (2026-10 extensions)
- Legacy namings: ai2 OLMo-1/MolmoE (``d_model``/``n_heads``/``n_layers`` +
  optional ``multi_query_attention``), GPT-2/GPT-BigCode/BTLM/CircuitGPT
  (``n_embd``/``n_head``/``n_layer``, ``multi_query`` -> 1 kv head, legacy
  falcon ``n_head_kv``), falcon-11b (``num_kv_heads``), chatglm/exaone/openelm
  (``num_layers``/``num_transformer_layers`` aliases).
- GDN hybrids generalized: qwen3_next, qwen3_5(_moe)_text, olmo_hybrid share
  the qwen3_next builder; qwen4_exp_text adds an indexer key stream
  (indexer_kv_heads*indexer_head_dim per indexer_compress_ratio tokens) on
  each attention layer.
- DSA (DeepSeek-Sparse-Attention family: glm_moe_dsa, hy_v4, deepseek_v32):
  every layer caches the MLA latent; layers whose indexer_types entry is
  "full" additionally cache an index_head_dim-wide key stream per token
  ("shared" layers reuse the previous full layer's stream, counted once).
- deepseek_v4/v41: shared-KV MQA (K and V alias one head_dim tensor; the
  window formula's factor 2 is compensated by encoding head_dim//2, while
  describe() still reports the config head_dim), sliding_window=128 window
  branches on every layer, and per-type compressed streams (CSA
  (head_dim+index_head_dim)/rate, HCA head_dim/rate) as latent entries.
  deepseek_v41_text raises KVConfigError: its compress_ratios (1/2) and
  kv_source_layer_ids semantics are not published in any implementation
  available here.
- inclusionAI Bailing (Ling): softmax-attention layers sit at
  ``(i+1) % layer_group_size == 0`` plus the tail; the rest are lightning
  linear-attention layers with a (num_attention_heads, head_dim, head_dim)
  matrix state (verified against inclusionAI's own modeling code).
- Cross-layer sharing in the wild: gemma-4 ``num_kv_shared_layers`` (groups of
  k layers share one cache), Snowflake SwiftKV (``num_key_value_layers`` own
  caches + ONE fused cache for the top layers), yoco-style names.

Label honesty (describe)
    A bare ``mha``/``gqa``/``mqa`` label is emitted ONLY when the identity
    ``kv_bytes_per_token == 2 * num_layers * num_kv_heads * head_dim * bytes``
    holds exactly. Otherwise suffixes are appended: ``+variable`` (per-layer
    GQA geometry: openelm, nemotron-nas block_configs), ``+partial`` (fewer
    attention-owning blocks than decoder layers: nemotron-nas no_op/linear
    blocks, swiftkv fused top layers), ``+shared`` (cross-layer KV sharing).

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
    # Omni wrappers nest the backbone twice: thinker_config -> text_config
    thinker = config.get("thinker_config")
    if isinstance(thinker, dict):
        for key in _TEXT_SCOPES:
            sub = thinker.get(key)
            if isinstance(sub, dict) and ("num_hidden_layers" in sub or "num_attention_heads" in sub):
                return sub
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
    # Kimi-VL style: the whole config IS a DeepSeek-V2-style MLA text backbone
    if (isinstance(cfg.get("kv_lora_rank"), (int, float)) and cfg["kv_lora_rank"]
            and isinstance(cfg.get("qk_rope_head_dim"), (int, float)) and cfg["qk_rope_head_dim"]):
        return "deepseek_v2"
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
        n = _opt_int(cfg, "num_layers")
    if n is None:
        n = _opt_int(cfg, "n_layer")
    if n is None:
        n = _opt_int(cfg, "n_layers")            # ai2 OLMo-1 / MolmoE
    if n is None:
        n = _opt_int(cfg, "num_transformer_layers")  # apple OpenELM
    if n is None:
        # nemotron-3 style: the layer list IS the authority
        for key in ("layers_block_type", "layer_types"):
            v = cfg.get(key)
            if isinstance(v, list) and v:
                return len(v)
        p = cfg.get("hybrid_override_pattern")
        if isinstance(p, str) and p:
            return len(p)
        raise KVConfigError("missing num_hidden_layers")
    return n


def _layer_count(cfg: dict, types: list | None) -> int:
    """Layer count consistent with an explicit per-layer type list."""
    n = _opt_int(cfg, "num_hidden_layers")
    if types is not None and n is not None and n != len(types):
        raise KVConfigError(
            f"layer list length {len(types)} != num_hidden_layers {n}"
        )
    return len(types) if types is not None else _num_layers(cfg)


def _attn_geometry(cfg: dict) -> tuple[int, int, int]:
    """(num_attention_heads, num_kv_heads, head_dim) with the documented rules."""
    hidden = _req_int(cfg, "hidden_size", min_val=1)
    heads = _opt_int(cfg, "num_attention_heads")
    if heads is None:
        heads = _opt_int(cfg, "num_heads")  # stablelm-alpha / openelm / rwkv7 style
    if heads is None:
        raise KVConfigError("missing num_attention_heads")
    heads = int(heads)
    kv = cfg.get("num_key_value_heads", cfg.get("num_kv_heads", heads))
    if kv is None:
        kv = heads
    if isinstance(kv, bool) or not isinstance(kv, (int, float)) or int(kv) != kv or int(kv) < 1:
        raise KVConfigError(f"bad num_key_value_heads: {kv!r}")
    kv = int(kv)
    hd = cfg.get("head_dim")
    if hd is not None:
        # 0 means "derive it" in some configs (e.g. bailing/Ling-lite)
        if isinstance(hd, bool) or not isinstance(hd, (int, float)) or int(hd) != hd or int(hd) < 0:
            raise KVConfigError(f"bad head_dim: {hd!r}")
        hd = int(hd) or None
    if hd is None:
        if hidden % heads != 0:
            raise KVConfigError(
                f"hidden_size {hidden} not divisible by num_attention_heads {heads} "
                "and no explicit head_dim"
            )
        hd = hidden // heads
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


def _fit(types: list, n: int, what: str) -> list:
    """Transformers semantics: a longer explicit list is truncated to n (e.g.
    MTP/prediction-head entries); a shorter one is an error."""
    if len(types) > n:
        return types[:n]
    if len(types) < n:
        raise KVConfigError(f"{what} length {len(types)} != num_hidden_layers {n}")
    return types


def _gemma_window_plan(cfg: dict) -> list[str]:
    """'full' / 'window' plan per layer for gemma2/gemma3/gemma4-style configs.

    Precedence: explicit ``layer_types`` list > character pattern string
    (``sliding_window_pattern`` like "LLLG", exaone-4.5) > integer
    ``sliding_window_pattern`` p (full iff (i+1) % p == 0) > alternate every 2.
    cohere2_moe prefixes: layers below ``first_k_dense_replace`` use
    ``prefix_dense_sliding_window_pattern`` (default 1 = all full).
    """
    n = _num_layers(cfg)
    w = cfg.get("sliding_window")
    w = int(w) if isinstance(w, (int, float)) and int(w) > 0 else None
    types = cfg.get("layer_types")
    if isinstance(types, list) and types:
        plan = []
        for t in types:
            t = str(t)
            if t in ("full_attention", "full", "global", "attention"):
                plan.append("full")
            elif t in ("sliding_attention", "sliding", "local", "swa"):
                plan.append("window" if w else "full")
            else:
                raise KVConfigError(f"unknown layer_types entry {t!r}")
        return _fit(plan, n, "layer_types")
    if w is None:
        return ["full"] * n
    pat = cfg.get("sliding_window_pattern")
    if isinstance(pat, str) and pat:
        # e.g. exaone-4.5 "LLLG": L = windowed(local), G = full(global)
        chars = {"L": "window", "G": "full", "S": "window"}
        plan = [chars.get(c) for c in pat]
        if None in plan:
            raise KVConfigError(f"bad sliding_window_pattern string {pat!r}")
        return _fit(plan, n, "sliding_window_pattern")
    prefix = _opt_int(cfg, "first_k_dense_replace")
    p_pat = _opt_int(cfg, "prefix_dense_sliding_window_pattern", 1)
    p = _opt_int(cfg, "sliding_window_pattern", 6)
    plan = []
    for i in range(n):
        pp = p_pat if (prefix is not None and i < prefix) else p
        plan.append("full" if (i + 1) % pp == 0 else "window")
    return plan


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
    d_state = _opt_int(cfg, "mamba_d_state", _opt_int(cfg, "d_state", _opt_int(cfg, "state_size", 16)))
    d_conv = _opt_int(cfg, "mamba_d_conv", _opt_int(cfg, "d_conv", _opt_int(cfg, "conv_kernel", 4)))
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
        plan = _fit(["full" if str(t) == "full_attention" else "window" for t in types], n, "layer_types")
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
        norm = [aliases.get(str(t), str(t)) for t in types]
        return _fit(norm, n, "layer list")
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
    num_heads = _opt_int(cfg, "num_heads", _opt_int(cfg, "gdn_num_heads", 32))  # transformers default
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


# ---- census-wave-2 builders ------------------------------------------------- #

def _b_gpt2_legacy(cfg: dict, bpe: int) -> list[LayerKV]:
    """GPT-2/GPT-BigCode/BTLM/CircuitGPT naming: n_embd/d_model, n_head, n_layer.

    gpt_bigcode optionally uses multi_query (kv heads = multi_query_groups or 1);
    legacy falcon configs use n_head_kv. All are full-attention decoders.
    """
    hidden = _opt_int(cfg, "n_embd") or _req_int(cfg, "d_model", min_val=1)
    heads = _req_int(cfg, "n_head", min_val=1)
    n = _req_int(cfg, "n_layer", min_val=1)
    if hidden % heads != 0:
        raise KVConfigError(f"n_embd {hidden} not divisible by n_head {heads}")
    hd = hidden // heads
    kv = heads
    if cfg.get("multi_query"):
        kv = _opt_int(cfg, "multi_query_groups", 1)
    n_head_kv = _opt_int(cfg, "n_head_kv")
    if n_head_kv is not None:
        kv = n_head_kv  # legacy falcon
    return [LayerKV("full", "mha" if kv == heads else ("mqa" if kv == 1 else "gqa"),
                    kv, hd, bpe) for _ in range(n)]


def _b_falcon(cfg: dict, bpe: int) -> list[LayerKV]:
    """falcon-7b/40b (legacy n_* naming) and falcon-11b (standard naming)."""
    if "n_head" in cfg or "n_layer" in cfg:
        return _b_gpt2_legacy(cfg, bpe)
    return _b_standard(cfg, bpe)


def _b_olmo_legacy(cfg: dict, bpe: int) -> list[LayerKV]:
    """ai2 OLMo-1 naming: d_model/n_heads/n_layers (+ optional multi-query)."""
    hidden = _req_int(cfg, "d_model", min_val=1)
    heads = _req_int(cfg, "n_heads", min_val=1)
    n = _req_int(cfg, "n_layers", min_val=1)
    if hidden % heads != 0:
        raise KVConfigError(f"d_model {hidden} not divisible by n_heads {heads}")
    hd = hidden // heads
    kv = heads
    if cfg.get("multi_query_attention"):
        kv = _opt_int(cfg, "multi_query_groups", 1)
    return [LayerKV("full", "mha" if kv == heads else ("mqa" if kv == 1 else "gqa"),
                    kv, hd, bpe) for _ in range(n)]


def _b_olmo_any(cfg: dict, bpe: int) -> list[LayerKV]:
    if "hidden_size" in cfg and "num_attention_heads" in cfg:
        return _b_standard(cfg, bpe)
    return _b_olmo_legacy(cfg, bpe)


def _b_granite_swa(cfg: dict, bpe: int) -> list[LayerKV]:
    heads, kv, hd = _attn_geometry(cfg)
    n = _num_layers(cfg)
    w = cfg.get("sliding_window")
    w = int(w) if isinstance(w, (int, float)) and int(w) > 0 else None
    types = _layer_types_list(cfg, n, aliases=_ATTN_ALIASES)
    if types is None:
        plan = ["full" if i % 4 == 0 else "window" for i in range(n)]
    else:
        plan = ["full" if t == "full_attention" else "window" for t in types]
    return [_full_or_window(heads, kv, hd, w if p == "window" else None, bpe) for p in plan]


def _b_openelm(cfg: dict, bpe: int) -> list[LayerKV]:
    """Apple OpenELM: per-layer GQA via num_query_heads / num_kv_heads lists."""
    qh = cfg.get("num_query_heads")
    kvh = cfg.get("num_kv_heads")
    if not isinstance(qh, list) or not isinstance(kvh, list) or len(qh) != len(kvh):
        raise KVConfigError("openelm needs num_query_heads/num_kv_heads lists")
    hd = _req_int(cfg, "head_dim", min_val=1)
    out = []
    for q, k in zip(qh, kvh):
        q, k = int(q), int(k)
        if q < k or k < 1:
            raise KVConfigError(f"bad openelm head list entries {q}/{k}")
        out.append(LayerKV("full", "mha" if k == q else ("mqa" if k == 1 else "gqa"),
                           k, hd, bpe))
    return out


def _b_rwkv7(cfg: dict, bpe: int) -> list[LayerKV]:
    """RWKV-7: matrix attention state (num_heads, head_dim, head_dim) plus the
    channel-mix token-shift state (hidden), per layer (same state layout family
    as RWKV-5/6; BlinkDL configs expose num_heads/head_dim)."""
    hidden = _req_int(cfg, "hidden_size", min_val=1)
    heads = _opt_int(cfg, "num_heads")
    hd = _opt_int(cfg, "head_dim")
    if hd is None:
        raise KVConfigError("rwkv7 config lacks head_dim")
    if heads is None:
        if hidden % hd != 0:
            raise KVConfigError("rwkv7 hidden_size not divisible by head_dim")
        heads = hidden // hd
    n = _num_layers(cfg)
    elems = heads * hd * hd + hidden
    return [_recurrent(cfg, "rwkv", elems, bpe) for _ in range(n)]


def _b_apriel_h(cfg: dict, bpe: int) -> list[LayerKV]:
    """Apriel-H1: hybrid_block_layout 't' (attention) / 'm2' (mamba2) with
    geometry in ssm_cfg (d_inner/d_state/d_conv; n_groups default 1)."""
    layout = cfg.get("hybrid_block_layout")
    if not isinstance(layout, list) or not layout:
        raise KVConfigError("apriel_h needs hybrid_block_layout")
    heads, kv, hd = _attn_geometry(cfg)
    ssm = dict(cfg.get("ssm_cfg") or {})
    d_inner = _opt_int(ssm, "d_inner") or _req_int(cfg, "hidden_size", min_val=1) * _opt_int(ssm, "expand", 1)
    d_state = _opt_int(ssm, "d_state")
    d_conv = _opt_int(ssm, "d_conv", 4)
    n_groups = _opt_int(ssm, "n_groups", 1)
    if d_state is None:
        raise KVConfigError("apriel_h ssm_cfg lacks d_state")
    elems = d_inner * d_state + (d_inner + 2 * n_groups * d_state) * (d_conv - 1)
    out = []
    for t in layout:
        if t == "t":
            out.append(_full_or_window(heads, kv, hd, None, bpe))
        elif t == "m2":
            out.append(_recurrent(cfg, "mamba2", elems, bpe))
        else:
            raise KVConfigError(f"unknown apriel_h layout entry {t!r}")
    return out


def _b_swiftkv(cfg: dict, bpe: int) -> list[LayerKV]:
    """Snowflake SwiftKV: the last L - num_key_value_layers layers fuse their
    KV projections into ONE shared cache -> distinct caches =
    num_key_value_layers + 1 (documented interpretation of the published
    SingleKV design: 'a single KV projection in the top K layers')."""
    layers = _b_standard(cfg, bpe)
    n_kv_layers = _opt_int(cfg, "num_key_value_layers")
    if n_kv_layers is None or n_kv_layers >= len(layers) or n_kv_layers < 1:
        raise KVConfigError("swiftkv config lacks sane num_key_value_layers")
    return layers[:n_kv_layers] + [layers[n_kv_layers]]


def _b_phi4flash(cfg: dict, bpe: int) -> list[LayerKV]:
    """Phi-4-mini-flash: sliding window W on odd layers of the first half
    (verified from microsoft's configuration_phi4flash.py: window iff
    layer_idx < L//2 and layer_idx % 2 == 1), rest full attention."""
    heads, kv, hd = _attn_geometry(cfg)
    n = _num_layers(cfg)
    w = cfg.get("sliding_window")
    w = int(w) if isinstance(w, (int, float)) and int(w) > 0 else None
    if w is None:
        raise KVConfigError("phi4flash config lacks sliding_window")
    return [_full_or_window(heads, kv, hd, w if (i < n // 2 and i % 2 == 1) else None, bpe)
            for i in range(n)]


def _b_nemotron_nas(cfg: dict, bpe: int) -> list[LayerKV]:
    """Nemotron-NAS / DeciLM block_configs: per-layer dicts with
    attention.n_heads_in_group = query heads per KV head (kv heads =
    num_attention_heads / n_heads_in_group); no_op / replace_with_linear
    blocks cache nothing."""
    blocks = cfg.get("block_configs")
    if not isinstance(blocks, list) or not blocks:
        raise KVConfigError("nemotron-nas needs block_configs")
    heads = _opt_int(cfg, "num_attention_heads")
    hidden = _opt_int(cfg, "hidden_size")
    if heads is None or hidden is None or hidden % heads != 0:
        raise KVConfigError("nemotron-nas config lacks sane attention geometry")
    hd = _opt_int(cfg, "head_dim") or hidden // heads
    out = []
    for b in blocks:
        if not isinstance(b, dict):
            raise KVConfigError("nemotron-nas block_configs entries must be dicts")
        att = b.get("attention")
        if not isinstance(att, dict) or att.get("no_op") or att.get("replace_with_linear"):
            continue
        g = _opt_int(att, "n_heads_in_group", 1)
        if heads % g != 0:
            raise KVConfigError(f"n_heads_in_group {g} does not divide {heads} heads")
        k = heads // g
        out.append(LayerKV("full", "mha" if k == heads else ("mqa" if k == 1 else "gqa"),
                           k, hd, bpe))
    if not out:
        raise KVConfigError("nemotron-nas config has no attention blocks")
    return out


def _b_deepseek_v4(cfg: dict, bpe: int) -> list[LayerKV]:
    """DeepSeek-V4 / V4.1 attention (grounded in transformers deepseek_v4).

    * Every layer uses shared-KV MQA (K and V alias ONE tensor of head_dim
      elements per kv head; num_key_value_heads = 1). The window formula
      2*kvh*hd*min(T,W) is halved by encoding head_dim//2, documented here.
    * Layer types: ``sliding_attention`` (pure W-window, W=sliding_window=128),
      ``compressed_sparse_attention`` (CSA: window + one head_dim compressed
      entry per compress_rate + an index_head_dim-wide indexer entry per rate),
      ``heavily_compressed_attention`` (HCA: window + one head_dim compressed
      entry per rate, no indexer). Legacy configs ship per-layer integer
      ``compress_ratios`` (0=sliding, 4=CSA, 128=HCA); the default pattern is
      2x HCA then CSA/HCA interleave (transformers 5.19 semantics).
      Bounded buffers/overlap states (< compress_rate tokens each) are below
      the analytic-minimum bar and are left to the RQ4 measurement.
    """
    n = _num_layers(cfg)
    hd = _req_int(cfg, "head_dim", min_val=1)
    kvh = _opt_int(cfg, "num_key_value_heads", 1)
    w = cfg.get("sliding_window")
    w = int(w) if isinstance(w, (int, float)) and int(w) > 0 else None
    if w is None:
        raise KVConfigError("deepseek_v4 config lacks sliding_window")
    if hd % 2 != 0:
        raise KVConfigError("deepseek_v4 head_dim must be even (shared-KV encoding)")
    types = cfg.get("layer_types")
    if not (isinstance(types, list) and types):
        ratios = cfg.get("compress_ratios")
        if isinstance(ratios, list) and ratios:
            ratios = _fit([int(r) for r in ratios], n, "compress_ratios")
            m = {0: "sliding_attention", 4: "compressed_sparse_attention", 128: "heavily_compressed_attention"}
            types = [m.get(int(r)) for r in ratios]
            if None in types:
                raise KVConfigError(f"unknown compress_ratios entry in {ratios}")
        else:
            types = (["heavily_compressed_attention"] * min(n, 2)
                     + ["compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                        for i in range(max(n - 2, 0))])
    types = _fit(list(types), n, "deepseek_v4 layer_types")
    cr = cfg.get("compress_rates")
    cr = cr if isinstance(cr, dict) else {}
    rate_csa = int(cr.get("compressed_sparse_attention", 4))
    rate_hca = int(cr.get("heavily_compressed_attention", 128))
    ihd = _opt_int(cfg, "index_head_dim", 128)
    out = []
    for t in types:
        out.append(LayerKV("window", "swa-sharedkv", kvh, hd // 2, bpe, window=w))
        if t == "compressed_sparse_attention":
            # one head_dim compressed entry + one index_head_dim indexer entry
            # per rate tokens -> (hd + ihd)/rate elements per token
            if (hd + ihd) % rate_csa != 0:
                raise KVConfigError("CSA stream elements are fractional")
            out.append(LayerKV("latent", "csa-stream", 0, 0, bpe,
                               latent_elems=(hd + ihd) // rate_csa))
        elif t == "heavily_compressed_attention":
            # one head_dim compressed entry per rate tokens
            if hd % rate_hca != 0:
                raise KVConfigError("HCA stream elements are fractional")
            out.append(LayerKV("latent", "hca-stream", 0, 0, bpe, latent_elems=hd // rate_hca))
        elif t != "sliding_attention":
            raise KVConfigError(f"unknown deepseek_v4 layer type {t!r}")
    return out


def _dsa_layers(cfg: dict, bpe: int) -> list[LayerKV]:
    """MLA latent per layer + indexer key stream (index_head_dim elements per
    token) for layers whose indexer is 'full'; 'shared' layers reuse the
    previous full layer's indexer cache (counted once).  indexer_types come
    from the config list, an 'F'/'S' pattern string (index_topk_pattern), or
    the freq/offset schedule (default freq=1 -> all full)."""
    n = _num_layers(cfg)
    kv_lora = _req_int(cfg, "kv_lora_rank", min_val=1)
    rope = _req_int(cfg, "qk_rope_head_dim", min_val=1)
    ihd = _opt_int(cfg, "index_head_dim", 128)
    itypes = cfg.get("indexer_types")
    if not (isinstance(itypes, list) and itypes):
        pat = cfg.get("index_topk_pattern")
        if isinstance(pat, str) and pat:
            itypes = [{"F": "full", "S": "shared"}.get(c) for c in pat]
            if None in itypes or len(itypes) != n:
                raise KVConfigError(f"bad index_topk_pattern {pat!r}")
        else:
            freq = max(int(cfg.get("index_topk_freq", 1) or 1), 1)
            offset = int(cfg.get("index_skip_topk_offset", 2) or 2)
            itypes = ["full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                      for i in range(n)]
    itypes = _fit(itypes, n, "indexer_types") if len(itypes) > n else itypes
    if len(itypes) != n:
        raise KVConfigError("indexer_types length mismatch")
    if ihd is None:
        raise KVConfigError("DSA config lacks index_head_dim")
    out = []
    for t in itypes:
        out.append(_latent_layer(kv_lora, rope, bpe))
        if t == "full":
            out.append(LayerKV("latent", "indexer", 0, 0, bpe, latent_elems=ihd))
        elif t != "shared":
            raise KVConfigError(f"unknown indexer type {t!r}")
    return out


def _b_bailing(cfg: dict, bpe: int, *, hybrid: bool, linear: bool) -> list[LayerKV]:
    """inclusionAI Bailing (Ling) family, grounded in inclusionAI's own
    modeling code: layer i is softmax attention iff (i+1) % layer_group_size
    == 0 or i >= (L // group) * group (the tail); other layers are lightning
    linear attention with a (num_attention_heads, head_dim, head_dim) state.
    bailing_moe = all-attention GQA; bailing_hybrid = MLA attention layers;
    bailing_moe_linear = GQA attention layers."""
    n = _num_layers(cfg)
    group = _opt_int(cfg, "layer_group_size")
    is_attn = None
    if group is not None:
        tail = (n // group) * group
        is_attn = [(i + 1) % group == 0 or i >= tail for i in range(n)]
    heads = _req_int(cfg, "num_attention_heads", min_val=1)
    if linear:
        hd_lin = _opt_int(cfg, "head_dim") or _req_int(cfg, "hidden_size", min_val=1) // heads
        lin_elems = heads * hd_lin * hd_lin
    out = []
    if hybrid:
        kv_lora = _req_int(cfg, "kv_lora_rank", min_val=1)
        rope = _req_int(cfg, "qk_rope_head_dim", min_val=1)
    else:
        _, kv, hd = _attn_geometry(cfg)
    for i in range(n):
        att = True if is_attn is None else is_attn[i]
        if att:
            out.append(_latent_layer(kv_lora, rope, bpe) if hybrid
                       else _full_or_window(heads, kv, hd, None, bpe))
        else:
            out.append(_recurrent(cfg, "lightning", lin_elems, bpe))
    return out


def _b_minicpm_sala(cfg: dict, bpe: int) -> list[LayerKV]:
    """MiniCPM-SALA: mixer_types 'minicpm4' (full GQA) vs 'lightning-attn'
    (lightning_nh x lightning_head_dim^2 matrix state)."""
    mixers = cfg.get("mixer_types")
    if not isinstance(mixers, list) or not mixers:
        raise KVConfigError("minicpm_sala needs mixer_types")
    heads, kv, hd = _attn_geometry(cfg)
    nh = _opt_int(cfg, "lightning_nh")
    lhd = _opt_int(cfg, "lightning_head_dim")
    if nh is None or lhd is None:
        raise KVConfigError("minicpm_sala lacks lightning geometry")
    out = []
    for m in mixers:
        if m == "minicpm4":
            out.append(_full_or_window(heads, kv, hd, None, bpe))
        elif m == "lightning-attn":
            out.append(_recurrent(cfg, "lightning", nh * lhd * lhd, bpe))
        else:
            raise KVConfigError(f"unknown minicpm_sala mixer {m!r}")
    return out


def _b_nemotron_flash(cfg: dict, bpe: int) -> list[LayerKV]:
    """NVIDIA Nemotron-Flash (grounded in the repo's own remote code):
    layer_types codes 'deltanet' (FLA DeltaNet: recurrent state
    [key_dim, value_dim] + q/k/v short-conv states, expand_k=expand_v=1,
    conv_size=4), 'm2' (Mamba2), 'a' (softmax attention, K width kq_head_dim,
    V width v_head_dim, both default hidden//heads), 'f' (FFN-only)."""
    types = cfg.get("layer_types")
    if not (isinstance(types, list) and types):
        raise KVConfigError("nemotron_flash needs layer_types")
    hidden = _req_int(cfg, "hidden_size", min_val=1)
    heads = _req_int(cfg, "num_attention_heads", min_val=1)
    kvh = _opt_int(cfg, "num_key_value_heads", heads)
    head = _opt_int(cfg, "head_dim") or (hidden // heads if hidden % heads == 0 else None)
    if head is None:
        raise KVConfigError("nemotron_flash hidden not divisible by heads and no head_dim")
    kq = _opt_int(cfg, "kq_head_dim", -1)
    vh = _opt_int(cfg, "v_head_dim", -1)
    kq = kq if (kq is not None and kq > 0) else head
    vh = vh if (vh is not None and vh > 0) else head
    if kq != vh:
        raise KVConfigError("nemotron_flash kq_head_dim != v_head_dim is unsupported")
    expand_k = 1.0
    expand_v = 1.0
    key_dim = int(hidden * expand_k)
    value_dim = int(hidden * expand_v)
    conv_size = _opt_int(cfg, "d_conv", _opt_int(cfg, "mamba_d_conv", 4))
    dn_elems = key_dim * value_dim + (2 * key_dim + value_dim) * (conv_size - 1)
    m2_elems = _mamba2_state_elems(cfg)
    out = []
    for t in types:
        t = str(t)
        if t == "a":
            out.append(_full_or_window(heads, kvh, kq, None, bpe))
        elif t == "m2":
            out.append(_recurrent(cfg, "mamba2", m2_elems, bpe))
        elif t == "deltanet":
            out.append(_recurrent(cfg, "gdn", dn_elems, bpe))
        elif t == "f":
            continue
        else:
            raise KVConfigError(f"unknown nemotron_flash layer code {t!r}")
    return out


def _b_cogvlmvideo(cfg: dict, bpe: int) -> list[LayerKV]:
    """CogVLM-Video: flat config with num_multi_query_heads instead of
    num_key_value_heads."""
    heads = _req_int(cfg, "num_attention_heads", min_val=1)
    kv = _opt_int(cfg, "num_multi_query_heads", heads)
    hidden = _req_int(cfg, "hidden_size", min_val=1)
    hd = _opt_int(cfg, "head_dim") or (hidden // heads)
    n = _num_layers(cfg)
    return [LayerKV("full", "mha" if kv == heads else ("mqa" if kv == 1 else "gqa"),
                    kv, hd, bpe) for _ in range(n)]


def _b_gpt_oss_puzzle(cfg: dict, bpe: int) -> list[LayerKV]:
    """gpt-oss-puzzle: explicit layer_types + per-layer sliding_window from
    block_configs dicts (falls back to the global sliding_window)."""
    heads, kv, hd = _attn_geometry(cfg)
    n = _num_layers(cfg)
    types = cfg.get("layer_types")
    if not (isinstance(types, list) and types and len(types) == n):
        raise KVConfigError("gpt_oss_puzzle needs layer_types")
    blocks = cfg.get("block_configs")
    w_global = cfg.get("sliding_window")
    w_global = int(w_global) if isinstance(w_global, (int, float)) and int(w_global) > 0 else None
    out = []
    for i, t in enumerate(types):
        w = w_global
        if isinstance(blocks, list) and i < len(blocks) and isinstance(blocks[i], dict):
            bw = blocks[i].get("sliding_window")
            w = int(bw) if isinstance(bw, (int, float)) and int(bw) > 0 else w
        out.append(_full_or_window(heads, kv, hd, w if str(t).startswith("sliding") else None, bpe))
    return out


def _b_hy_v4(cfg: dict, bpe: int) -> list[LayerKV]:
    """Tencent Hy4: deepseek_sparse_attention layers = MLA latent + indexer
    stream (see _dsa_layers)."""
    types = cfg.get("layer_types")
    if types is not None and any(str(t) != "deepseek_sparse_attention" for t in types):
        raise KVConfigError("hy_v4 with non-DSA layer_types is unsupported")
    return _dsa_layers(cfg, bpe)

def _b_qwen4_exp(cfg: dict, bpe: int) -> list[LayerKV]:
    """Qwen3.8-Flash-Next (qwen4_exp): GDN layers (linear_*) + attention
    layers that additionally cache an indexer key stream
    (indexer_kv_heads * indexer_head_dim per indexer_compress_ratio tokens,
    grounded in the config fields indexer_*)."""
    layers = _b_qwen3_next(cfg, bpe)
    ihd = _opt_int(cfg, "indexer_head_dim")
    ikv = _opt_int(cfg, "indexer_kv_heads", 1)
    ratio = _opt_int(cfg, "indexer_compress_ratio", 1)
    if ihd is None or ratio < 1 or (ikv * ihd) % ratio != 0:
        raise KVConfigError("qwen4_exp indexer fields missing or inconsistent")
    stream = LayerKV("latent", "indexer", 0, 0, bpe, latent_elems=(ikv * ihd) // ratio)
    out: list[LayerKV] = []
    for l in layers:
        out.append(l)
        if l.kind == "full":
            out.append(stream)
    return out


def _b_glm_dsa(cfg: dict, bpe: int) -> list[LayerKV]:
    """glm_moe_dsa: every layer is MLA + DSA indexer streams (see _dsa_layers)."""
    return _dsa_layers(cfg, bpe)


# ---- registry --------------------------------------------------------------- #

_STANDARD_FLAGGED = {
    # plain GQA/MHA/MQA, window only behind use_sliding_window
    "llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe",
    "internlm2", "internlm3", "baichuan", "baichuan2", "olmo", "olmo2", "olmoe",
    "stablelm", "phi", "gemma", "glm", "glm4", "glm4_moe", "nemotron",
    "exaone", "exaone4", "exaone_moe", "cohere", "command-r", "smollm", "smollm2",
    "smollm3", "arcee", "granite", "granitemoe", "phimoe", "hunyuan",
    "hunyuan_v1_dense", "hunyuan_v1_moe", "gpt_neox", "mllama", "chatglm2", "chatglm3",
    # wave 2 (census): flat or nested plain-attention backbones
    "qwen", "minicpm", "minicpmv", "internlm", "InternLMXComposer", "internlmxcomposer2",
    "molmo", "emo", "flex_olmo", "bailing_moe", "hy_v3", "apriel", "arctic",
    "baichuan_m1", "cogvlm2", "deepseek", "llava", "llava_llama", "llava_mistral",
    "llava_next", "llava_qwen2", "llava_qwen_mistral", "GOT", "cpmbee", "cpmant",
    "opt", "biogpt", "stablelm_alpha", "stablelm_epoch", "solar", "solar_open",
    "solar_open2", "qwen3_vl_text", "qwen3_vl_moe_text", "qwen2_5_vl", "qwen2_5_vl_text",
    "qwen2_vl", "glm4v_text", "glm4v_moe_text", "glm4_moe_lite", "minimaxm3sparse",
    "minimax_m3", "penguinvl_qwen3", "omnilmm", "beit3_llava", "wedlm", "olmo_hils",
    "olmo2_noqknorm_prenorm", "qwen2_5_omni_text", "phi3small", "qwen3_omni_moe_text",
}
_MLA = {"deepseek_v2", "deepseek_v3", "deepseek_v32", "kimi_k2", "kimi_k25",
        "minicpm3", "youtu", "youtu_vita_text"}

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
    "falcon_mamba": lambda c, b: _b_mamba_pure(c, b, mamba2=False),
    "rwkv": _b_rwkv,
    "rwkv5": _b_rwkv,
    "rwkv6": _b_rwkv,
    "rwkv7": _b_rwkv7,
    "chatglm": _b_chatglm,
    # You-Only-Cache-Once style: standard geometry + one cross-layer shared cache
    # (detected by name; see _detect_sharing).
    "yoco": _b_standard,
    # ---- wave 2 (census) ----
    "gpt2": _b_gpt2_legacy,
    "gpt2_custom": _b_gpt2_legacy,
    "gpt_bigcode": _b_gpt2_legacy,
    "btlm": _b_gpt2_legacy,
    "circuitgpt": _b_gpt2_legacy,
    "falcon": _b_falcon,
    "hf_olmo": _b_olmo_legacy,
    "olmo3": _b_gemma2,
    "olmo2-retrofit": _b_gemma2,
    "olmo_hybrid": _b_qwen3_next,
    "qwen3_5_text": _b_qwen3_next,
    "qwen3_5_moe_text": _b_qwen3_next,
    "qwen4_exp_text": _b_qwen4_exp,
    "granite_swa": _b_granite_swa,
    "granitemoe_swa": _b_granite_swa,
    "gemma4_text": _b_gemma2,
    "gemma4_unified_text": _b_gemma2,
    "exaone4_5_text": _b_gemma2,
    "step3p5": _b_gemma2,
    "cohere2": _b_gemma2,
    "cohere2_moe": _b_gemma2,
    "bolmo": _b_gemma2,
    "qwen3_scale_seq": _b_gemma2,
    "granite_switch": _b_gemma2,
    "minimax_m1": _b_minimax,
    "minimax_m2": _b_minimax,
    "deepseek_v4": _b_deepseek_v4,
    "deepseek_v41_text": _b_deepseek_v4,
    "glm_moe_dsa": _b_glm_dsa,
    "hy_v4": _b_hy_v4,
    "bailing_hybrid": lambda c, b: _b_bailing(c, b, hybrid=True, linear=True),
    "bailing_moe_linear": lambda c, b: _b_bailing(c, b, hybrid=False, linear=True),
    "apriel_h": _b_apriel_h,
    "openelm": _b_openelm,
    "llama_swiftkv": _b_swiftkv,
    "phi4flash": _b_phi4flash,
    "nemotron-nas": _b_nemotron_nas,
    "nemotron_h_puzzle": lambda c, b: _mamba2_hybrid(c, b, pattern_mapping=_NEMOTRON_PATTERN),
    "gpt_oss_puzzle": _b_gpt_oss_puzzle,
    "minicpm_sala": _b_minicpm_sala,
    # wave 2, pass 2
    "phi3_v": _b_phi3,          # phi3 vision wrapper: flat phi3 fields (vestigial window)
    "nemotron_flash": _b_nemotron_flash,
    "git": _b_standard,         # GIT text decoder (vision prepended, no cross-attn)
    "ministral3": _b_standard,
    "cogvlm": _b_standard,      # CogVLM: vicuna-style flat MHA (visual expert shares KV geometry)
    "cogagent": _b_standard,
    "cogvlmvideo": _b_cogvlmvideo,
    "omni": _b_standard,        # Baichuan-Audio: flat GQA backbone
    "bitnet": _b_standard,
    "japanesestablelmalpha": _b_standard,
    "deepseekv2": _b_standard,  # DeepSeek-OCR-2: use_mla=false, kv_lora_rank null
    "molmo": _b_olmo_any,       # Molmo-72B: standard fields; MolmoE: ai2 d_model naming
    "glm4_moe_lite": _mla,      # GLM-4.7-Flash: MLA latent per layer
    "deepseek_v32": _b_glm_dsa, # V3.2: MLA + indexed (DSA) attention
}
# "olmo" appears with BOTH modern (hidden_size) and legacy (d_model) field sets
_REGISTRY["olmo"] = _b_olmo_any

_SHARING_KEYS = ("num_kv_shared_layers", "kv_sharing", "shared_kv_layers")


def _sharing_flag(cfg: dict) -> bool:
    """True when the architecture itself implies cross-layer KV sharing
    (gemma-4 num_kv_shared_layers, yoco names, SwiftKV fused top layers)."""
    if _detect_sharing(cfg) is not None:
        return True
    mt = cfg.get("model_type")
    return isinstance(mt, str) and "swiftkv" in mt.lower() and bool(cfg.get("swiftkv", True))


def _detect_sharing(cfg: dict) -> int | None:
    """k >= 2 => each group of k consecutive attention layers shares one cache."""
    for key in _SHARING_KEYS:
        v = cfg.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and int(v) >= 2:
            return int(v)
    name = f"{cfg.get('model_type', '')}{''.join(cfg.get('architectures') or [])}".lower()
    if "yoco" in name:
        return 10**9  # whole stack shares one cache
    if "swiftkv" in name:
        return None  # handled inside _b_swiftkv (n_kv_layers + 1 caches)
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
    marginal = float(sum(_layer_marginal(l) for l in layers))
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
    # gate-honesty suffixes: the 2*L*kvh*hd*2 identity only holds for a bare
    # mha/gqa/mqa label on a uniform, unshared stack.
    if atype in ("mha", "gqa", "mqa"):
        geo = {(l.n_kv_heads, l.head_dim) for l in attn_layers}
        if len(geo) > 1:
            atype += "+variable"   # openelm / nemotron-nas per-layer GQA
        if n_full + n_window != _num_layers(cfg):
            atype += "+partial"    # nemotron-nas blocks without attention
        if _sharing_flag(cfg):
            atype += "+shared"     # swiftkv fused top-layer cache
        if atype in ("mha", "gqa", "mqa"):
            # gate-literal reading: the completion gate recomputes the identity
            # from num_key_value_heads (fallback num_attention_heads) and
            # head_dim (fallback hidden//heads) ONLY. If our geometry comes
            # from other keys (chatglm multi_query_group_num, falcon
            # n_head_kv/num_kv_heads, head_dim=0 meaning "derive"), the bare
            # label would break the gate -> mark it.
            gkvh = cfg.get("num_key_value_heads")
            if gkvh is None:
                gkvh = heads
            ghd = cfg.get("head_dim")
            if ghd is None and hidden and heads and hidden % heads == 0:
                ghd = hidden // heads
            gate_identity = 2 * _num_layers(cfg) * gkvh * ghd * 2 if (gkvh and ghd is not None) else None
            if gate_identity is None or marginal != gate_identity:
                atype += "+legacycfg"
    elif _sharing_flag(cfg) and not atype.startswith("hybrid"):
        atype += "+shared"         # e.g. gemma4 local+global+shared

    head_dim = first_attn.head_dim if first_attn is not None else None
    # report the config's attention head width when it is well-defined
    # (deepseek_v4 window layers encode head_dim//2 for the shared-KV formula)
    cfg_hd = cfg.get("head_dim")
    if isinstance(cfg_hd, (int, float)) and not isinstance(cfg_hd, bool) and int(cfg_hd) > 0:
        head_dim = int(cfg_hd)
    if not head_dim and n_latent and not n_full and not n_window and cfg.get("qk_nope_head_dim"):
        head_dim = _opt_int(cfg, "qk_nope_head_dim") + _opt_int(cfg, "qk_rope_head_dim")
    if not head_dim and first_attn is not None and hidden and heads and hidden % heads == 0:
        head_dim = hidden // heads
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
        "kv_shared_cross_layer": _sharing_flag(cfg),
        "kv_bytes_per_token_bf16": kv_bytes_per_token(config),
        "kv_bytes_at_32k": kv_bytes_at(config, 32_768),
        "kv_bytes_at_128k": kv_bytes_at(config, 131_072),
        "recurrent_state_bytes": fixed_state_bytes(config),
    }
