"""Unit tests for analysis/kv.py — analytic KV accounting.

All expected values are HAND-COMPUTED; the arithmetic is shown in comments.
Real-config fixtures under tests/fixtures/raw_configs/ were fetched from the
Hugging Face Hub (see FETCHED.md in that directory for provenance; gated
families use the mirrors noted there).  Run with `make test`.
"""

import json
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "analysis"))
import kv  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "raw_configs"


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


# --------------------------------------------------------------------------- #
# synthetic configs (hand-computed expectations)
# --------------------------------------------------------------------------- #

def llama_like(L=32, heads=32, kv=8, hidden=4096, head_dim=None, **extra):
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": L,
        "num_attention_heads": heads,
    }
    if kv is not None:
        cfg["num_key_value_heads"] = kv
    if head_dim is not None:
        cfg["head_dim"] = head_dim
    cfg.update(extra)
    return cfg


class TestFullAttention:
    def test_pure_mha(self):
        # gpt-neox style, no num_key_value_heads -> kv = heads (MHA)
        # hd = 4096/32 = 128; per-token = 2*32*128*2 = 16384 B; 12 layers -> 196608
        cfg = llama_like(L=12, heads=32, kv=None)
        assert kv.kv_bytes_per_token(cfg) == 2 * 12 * 32 * 128 * 2  # = 196_608
        # at T=1000: 196608 * 1000 = 196_608_000
        assert kv.kv_bytes_at(cfg, 1000) == 196608 * 1000
        assert kv.describe(cfg)["attention_type"] == "mha"
        assert kv.fixed_state_bytes(cfg) == 0.0

    def test_pure_gqa(self):
        # kv=8 of 32 heads: per-token = 2*8*128*2 = 4096 B; 28 layers -> 114_688
        cfg = llama_like(L=28, heads=32, kv=8)
        assert kv.kv_bytes_per_token(cfg) == 2 * 28 * 8 * 128 * 2  # = 114_688
        d = kv.describe(cfg)
        assert d["attention_type"] == "gqa"
        assert d["gqa_ratio"] == pytest.approx(4.0)
        assert d["n_full_layers"] == 28 and d["n_window_layers"] == 0

    def test_mqa(self):
        # kv=1: per-token = 2*1*128*2 = 512 B; 24 layers -> 12_288
        cfg = llama_like(L=24, heads=32, kv=1)
        assert kv.kv_bytes_per_token(cfg) == 2 * 24 * 1 * 128 * 2  # = 12_288
        assert kv.describe(cfg)["attention_type"] == "mqa"

    def test_explicit_head_dim_beats_hidden_over_heads(self):
        # gemma-1 style: hidden 2048, 8 heads -> 256 == head_dim; make it differ:
        # hidden 1024, 8 heads (hidden//heads = 128) but explicit head_dim = 256.
        # per-token = 2*1*256*2 = 1024 B; 18 layers -> 18_432
        cfg = llama_like(L=18, heads=8, kv=1, hidden=1024, head_dim=256)
        assert kv.kv_bytes_per_token(cfg) == 2 * 18 * 1 * 256 * 2  # = 18_432
        assert kv.describe(cfg)["head_dim"] == 256

    def test_bytes_per_element_fp8(self):
        # bpe=1 halves every number: 2*4*8*128*1 = 8192
        cfg = llama_like(L=4, heads=32, kv=8)
        assert kv.kv_bytes_per_token(cfg, bytes_per_element=1) == 2 * 4 * 8 * 128 * 1


class TestSlidingWindow:
    def test_mistral_style_flag_on_all_window(self):
        # use_sliding_window=true, W=4096: every layer is a window layer.
        # asymptotic marginal = 0; at T=1000 (< W) each layer holds 1000 tokens.
        cfg = llama_like(L=8, heads=32, kv=8, use_sliding_window=True, sliding_window=4096)
        assert kv.kv_bytes_per_token(cfg) == 0.0
        assert kv.kv_bytes_at(cfg, 1000) == 8 * 2 * 8 * 128 * 2 * 1000
        d = kv.describe(cfg)
        assert d["attention_type"] == "gqa+swa"
        assert d["n_window_layers"] == 8 and d["n_full_layers"] == 0
        assert d["window_size"] == 4096

    def test_T_gt_W_saturates(self):
        # W=512, T=4096: each window layer holds min(4096,512)=512 tokens
        # -> total = 8 * 2*8*128*2 * 512 = 8_388_608
        cfg = llama_like(L=8, heads=32, kv=8, use_sliding_window=True, sliding_window=512)
        assert kv.kv_bytes_at(cfg, 4096) == 8 * 2 * 8 * 128 * 2 * 512  # = 8_388_608
        assert kv.kv_bytes_at(cfg, 512) == 8 * 2 * 8 * 128 * 2 * 512
        assert kv.kv_bytes_at(cfg, 511) == 8 * 2 * 8 * 128 * 2 * 511

    def test_flag_off_means_full(self):
        # qwen2.5 style: sliding_window present but use_sliding_window=false
        cfg = llama_like(L=4, heads=32, kv=4, use_sliding_window=False, sliding_window=131072)
        assert kv.kv_bytes_per_token(cfg) == 2 * 4 * 4 * 128 * 2
        assert kv.describe(cfg)["n_window_layers"] == 0

    def test_phi3_window_is_vestigial(self):
        # phi3 carries sliding_window=2047 but attention is full causal
        cfg = llama_like(L=4, heads=32, kv=32, model_type="phi3", sliding_window=2047)
        assert kv.kv_bytes_per_token(cfg) == 2 * 4 * 32 * 128 * 2
        assert kv.describe(cfg)["n_window_layers"] == 0

    def test_gemma_pattern(self):
        # gemma-2/3: sliding_window_pattern=6 -> layer i global iff (i+1)%6==0.
        # L=12, kv=1, hd=256: full layers i=5,11 (2), window layers 10.
        # asymptotic = 2 * 2*1*256*2 = 2048 B/token.
        # at T=100000 (>W=512): windows hold 512 -> 10*1024*512 + 2*1024*100000
        cfg = llama_like(L=12, heads=4, kv=1, hidden=1024, head_dim=256,
                         model_type="gemma3_text", sliding_window=512,
                         sliding_window_pattern=6)
        assert kv.kv_bytes_per_token(cfg) == 2 * 2 * 1 * 256 * 2  # = 2_048
        assert kv.kv_bytes_at(cfg, 100000) == 10 * 2 * 1 * 256 * 2 * 512 + 2 * 2 * 1 * 256 * 2 * 100000
        d = kv.describe(cfg)
        assert (d["n_full_layers"], d["n_window_layers"]) == (2, 10)
        assert d["attention_type"] == "local+global"
        assert d["window_size"] == 512

    def test_gemma_explicit_layer_types(self):
        # explicit layer_types list wins: alternate sliding/full 1:1 for L=6
        types = ["sliding_attention", "full_attention"] * 3
        cfg = llama_like(L=6, heads=4, kv=1, hidden=1024, head_dim=256,
                         model_type="gemma2", sliding_window=4096, layer_types=types)
        d = kv.describe(cfg)
        assert (d["n_full_layers"], d["n_window_layers"]) == (3, 3)

    def test_starcoder2_window_without_flag(self):
        # starcoder2 applies sliding_window with no flag
        cfg = llama_like(L=4, heads=24, kv=2, hidden=3072, model_type="starcoder2", sliding_window=4096)
        assert kv.kv_bytes_per_token(cfg) == 0.0
        assert kv.describe(cfg)["n_window_layers"] == 4

    def test_gpt_oss(self):
        # gpt-oss-20b geometry, synthetic: 24 layers alternating sliding(W=128)/full,
        # 64 heads, 8 kv, hd 64 -> per full layer 2*8*64*2 = 2048 B/token.
        types = ["sliding_attention", "full_attention"] * 12
        cfg = {
            "model_type": "gpt_oss", "hidden_size": 2880, "num_hidden_layers": 24,
            "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64,
            "sliding_window": 128, "layer_types": types,
        }
        assert kv.kv_bytes_per_token(cfg) == 12 * 2 * 8 * 64 * 2  # = 24_576
        # at T=1_000_000: full layers grow, window layers capped at 128
        assert kv.kv_bytes_at(cfg, 1_000_000) == 12 * 2048 * 1_000_000 + 12 * 2048 * 128
        d = kv.describe(cfg)
        assert d["attention_type"] == "local+global"
        assert (d["n_full_layers"], d["n_window_layers"]) == (12, 12)


class TestMLA:
    def test_deepseek_v2_all_layers_latent(self):
        # DeepSeek-V2 style: latent = (kv_lora_rank + qk_rope_head_dim) elems/token.
        # 512 + 64 = 576 elems; BF16 -> 1152 B/token/layer; 27 layers -> 31_104.
        # first_k_dense_replace affects the FFN (dense vs MoE), NOT attention:
        # all layers stay latent.
        cfg = {
            "model_type": "deepseek_v2", "hidden_size": 2048, "num_hidden_layers": 27,
            "num_attention_heads": 16, "num_key_value_heads": 16,
            "kv_lora_rank": 512, "qk_rope_head_dim": 64, "qk_nope_head_dim": 128,
            "v_head_dim": 128, "first_k_dense_replace": 1,
        }
        layers = kv.parse_layers(cfg)
        assert len(layers) == 27 and all(l.kind == "latent" for l in layers)
        assert all(l.latent_elems == 576 for l in layers)
        assert kv.kv_bytes_per_token(cfg) == 27 * 576 * 2  # = 31_104
        assert kv.kv_bytes_at(cfg, 128) == 27 * 576 * 2 * 128
        d = kv.describe(cfg)
        assert d["attention_type"] == "mla"
        assert d["n_latent_layers"] == 27
        assert d["kv_lora_rank"] == 512 and d["qk_rope_head_dim"] == 64
        assert d["head_dim"] == 128 + 64  # qk_nope + qk_rope

    def test_mla_at_finite_T_and_par(self):
        cfg = {
            "model_type": "deepseek_v2", "hidden_size": 2048, "num_hidden_layers": 4,
            "num_attention_heads": 16, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
        }
        # 4 * 1152 = 4608 B/token; parity with 1e9 params * 2 B = 2e9 bytes
        # T* = ceil(2e9 / 4608) = 434_028 (check exactness below)
        target = 1e9 * 2
        T = kv.kv_par_context(cfg, total_params=1e9)
        assert kv.kv_bytes_at(cfg, int(T)) >= target
        assert kv.kv_bytes_at(cfg, int(T) - 1) < target
        assert T == math.ceil(target / (4 * 576 * 2))


class TestHybridRecurrent:
    def test_granite_style_mamba2_hybrid(self):
        # granite-4.0-h style: layer_types with mamba/attention; mamba2 params.
        # hidden 1024, expand 2 -> d_inner 2048 (= 32 heads * 64);
        # ssm = 2048*128 = 262_144 elems; conv_dim = 2048 + 2*1*128 = 2304,
        # conv = 2304*(4-1) = 6_912 elems; state elems = 269_056 -> 538_112 B.
        # 3 mamba layers -> 1_614_336 B fixed.
        # attention: 2 kv heads, hd 64 -> 2*2*64*2 = 512 B/token per layer.
        cfg = {
            "model_type": "granitemoehybrid", "hidden_size": 1024, "num_hidden_layers": 4,
            "num_attention_heads": 16, "num_key_value_heads": 2,
            "layer_types": ["mamba", "attention", "mamba", "attention"],
            "mamba_n_heads": 32, "mamba_d_head": 64, "mamba_d_state": 128,
            "mamba_n_groups": 1, "mamba_d_conv": 4, "mamba_expand": 2,
        }
        layers = kv.parse_layers(cfg)
        kinds = [l.kind for l in layers]
        assert kinds == ["recurrent", "full", "recurrent", "full"]
        per_state = (2048 * 128 + (2048 + 2 * 1 * 128) * 3) * 2  # = 538_112 B
        assert kv.fixed_state_bytes(cfg) == 2 * per_state        # = 1_076_224
        assert kv.kv_bytes_per_token(cfg) == 2 * (2 * 2 * 64 * 2)  # = 1_024
        # hybrid total at T: full-KV part + constant state
        assert kv.kv_bytes_at(cfg, 1000) == 2 * 512 * 1000 + 2 * per_state
        assert kv.describe(cfg)["attention_type"] == "hybrid-mamba"

    def test_jamba_style_mamba1(self):
        # jamba: layers_block_type attention/mamba; mamba1 state has conv over
        # d_inner channels only: ssm = d_inner*d_state, conv = d_inner*(d_conv-1).
        # hidden 512, expand 2 -> d_inner 1024; d_state 16, d_conv 4:
        # ssm 16_384 + conv 3_072 = 19_456 elems -> 38_912 B per mamba layer.
        cfg = {
            "model_type": "jamba", "hidden_size": 512, "num_hidden_layers": 4,
            "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64,
            "layers_block_type": ["attention", "mamba", "mamba", "mamba"],
            "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2,
        }
        assert kv.fixed_state_bytes(cfg) == 3 * 19_456 * 2  # = 116_736
        assert kv.kv_bytes_per_token(cfg) == 2 * 2 * 64 * 2  # = 512 (1 attention layer)
        d = kv.describe(cfg)
        assert (d["n_full_layers"], d["n_recurrent_layers"]) == (1, 3)

    def test_jamba_derived_pattern(self):
        # no layers_block_type: attention iff i % 8 == 4 -> layers 4,12 of 16
        cfg = {
            "model_type": "jamba", "hidden_size": 512, "num_hidden_layers": 16,
            "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64,
            "attn_layer_period": 8, "attn_layer_offset": 4,
            "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2,
        }
        d = kv.describe(cfg)
        assert (d["n_full_layers"], d["n_recurrent_layers"]) == (2, 14)

    def test_falcon_h1_every_layer_attention_plus_mamba(self):
        # falcon-h1: EACH layer owns both a GQA KV and a mamba2 state.
        # hidden 512, expand 2, mamba_n_heads 8, d_head 64 -> d_inner 512;
        # d_state 256, n_groups 1, d_conv 4:
        # ssm = 512*256 = 131_072; conv_dim = 512 + 2*256 = 1024, conv = 3072;
        # elems = 134_144 -> 268_288 B per layer.
        # attention kv=2, hd=64: 2*2*64*2 = 512 B/token per layer.
        cfg = {
            "model_type": "falcon_h1", "hidden_size": 512, "num_hidden_layers": 6,
            "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64,
            "mamba_n_heads": 8, "mamba_d_head": 64, "mamba_d_state": 256,
            "mamba_n_groups": 1, "mamba_d_conv": 4, "mamba_expand": 2,
        }
        layers = kv.parse_layers(cfg)
        assert len(layers) == 12  # 6 attention + 6 recurrent units
        assert [l.kind for l in layers[:2]] == ["full", "recurrent"]
        assert kv.kv_bytes_per_token(cfg) == 6 * 512  # = 3_072
        assert kv.fixed_state_bytes(cfg) == 6 * 268_288  # = 1_609_728
        assert kv.kv_bytes_at(cfg, 500) == 6 * 512 * 500 + 6 * 268_288

    def test_nemotron_h_pattern_letters(self):
        # nemotron_h hybrid_override_pattern: M=mamba, E=moe(no cache),
        # *=full attention, -=mlp(no cache).  Pattern "MM*ME-M" (6 blocks):
        # 1 attention, 3 mamba, 2 FFN-only.  mamba2: hidden 512, expand 2
        # -> d_inner 1024; d_state 128, n_groups 8, conv 4:
        # ssm = 131_072; conv_dim = 1024 + 2*8*128 = 3072; conv = 9_216;
        # elems = 140_288 -> 280_576 B per mamba layer.
        cfg = {
            "model_type": "nemotron_h", "hidden_size": 512, "num_hidden_layers": 6,
            "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64,
            "hybrid_override_pattern": "MM*ME-",
            "mamba_num_heads": 16, "mamba_head_dim": 64, "ssm_state_size": 128,
            "n_groups": 8, "conv_kernel": 4, "expand": 2,
        }
        layers = kv.parse_layers(cfg)
        kinds = [l.kind for l in layers]
        # pattern MM*ME-: M,M -> recurrent; * -> full; M -> recurrent;
        # E (moe) and - (mlp) are FFN-only mixers -> no cache unit at all.
        assert kinds == ["recurrent", "recurrent", "full", "recurrent"]
        assert kv.fixed_state_bytes(cfg) == 3 * 280_576
        assert kv.kv_bytes_per_token(cfg) == 1 * 2 * 2 * 64 * 2  # = 512
        d = kv.describe(cfg)
        assert (d["n_full_layers"], d["n_recurrent_layers"]) == (1, 3)


class TestPureRecurrent:
    def test_mamba_original_format(self):
        # state-spaces/mamba-2.8b format: d_model 512, n_layer 4, defaults
        # expand 2, d_state 16, d_conv 4 -> d_inner 1024:
        # ssm = 1024*16 = 16_384; conv = 1024*3 = 3_072; elems 19_456 -> 38_912 B
        cfg = {"d_model": 512, "n_layer": 4, "ssm_cfg": {}}
        assert kv.kv_bytes_per_token(cfg) == 0.0
        assert kv.fixed_state_bytes(cfg) == 4 * 19_456 * 2  # = 155_648
        assert kv.kv_bytes_at(cfg, 100000) == 4 * 19_456 * 2  # T-independent
        d = kv.describe(cfg)
        assert d["attention_type"] == "mamba"
        assert d["num_kv_heads"] is None

    def test_mamba2_original_format(self):
        # state-spaces/mamba2 format: d_model 512, ssm_cfg.layer=Mamba2,
        # defaults expand 2, d_state 128, n_groups 8, headdim 64 -> d_inner 1024:
        # ssm = 1024*128 = 131_072; conv_dim = 1024 + 2*8*128 = 3_072;
        # conv = 3_072*3 = 9_216; elems 140_288 -> 280_576 B per layer
        cfg = {"d_model": 512, "n_layer": 3, "ssm_cfg": {"layer": "Mamba2"}}
        assert kv.fixed_state_bytes(cfg) == 3 * 280_576  # = 841_728

    def test_falcon_mamba(self):
        # falcon-mamba-7b: mamba1-style with expand 16, state_size 16,
        # conv_kernel 4: d_inner = 16*4096 = 65_536;
        # ssm = 65_536*16 = 1_048_576; conv = 65_536*3 = 196_608 elems
        # -> 1_245_184 elems -> 2_490_368 B per layer, 64 layers.
        cfg = load("tiiuae__falcon-mamba-7b")
        assert kv.fixed_state_bytes(cfg) == 64 * 1_245_184 * 2  # = 159_383_552
        assert kv.kv_bytes_per_token(cfg) == 0.0
        assert kv.describe(cfg)["attention_type"] == "mamba"

    def test_recurrent_gemma(self):
        # block_types ["recurrent","recurrent","attention"], hidden 2560,
        # lru_width 2560, conv1d_width 4: rnn = 2560, conv = 2560*3 = 7_680
        # -> 10_240 elems -> 20_480 B per recurrent layer.
        # attention: kv=1, hd 256, window 2048 -> 2*1*256*2 = 1_024 B/token.
        # L=6: pattern idx 2 and 5 are attention -> 2 window + 4 recurrent.
        cfg = {
            "model_type": "recurrent_gemma", "hidden_size": 2560, "num_hidden_layers": 6,
            "num_attention_heads": 10, "num_key_value_heads": 1, "head_dim": 256,
            "attention_window_size": 2048, "conv1d_width": 4, "lru_width": 2560,
            "block_types": ["recurrent", "recurrent", "attention"],
        }
        assert kv.fixed_state_bytes(cfg) == 4 * 20_480  # = 81_920
        assert kv.kv_bytes_per_token(cfg) == 0.0  # all attention is windowed
        assert kv.kv_bytes_at(cfg, 4096) == 2 * 1024 * 2048 + 4 * 20_480
        d = kv.describe(cfg)
        assert d["attention_type"] == "hybrid-lru"
        assert (d["n_window_layers"], d["n_recurrent_layers"]) == (2, 4)

    def test_rwkv4(self):
        # rwkv-4: 5 * hidden elements per layer -> 5*512*2 = 5_120 B
        cfg = {"model_type": "rwkv", "hidden_size": 512, "num_hidden_layers": 24}
        assert kv.fixed_state_bytes(cfg) == 24 * 5 * 512 * 2

    def test_lfm2(self):
        # LFM2: full_attn_idxs [1] (0-based) of 3 layers; gdn heads 32 of 64-wide
        # hidden: rnn = 32*(64*64) = 131_072; conv = conv_dim 64 * (3-1) = 128
        # elems -> 131_200 * 2 = 262_400 B per gdn layer.
        # attention kv=4, hd 64 -> 2*4*64*2 = 1_024 B/token.
        cfg = {
            "model_type": "lfm2", "hidden_size": 2048, "num_hidden_layers": 3,
            "num_attention_heads": 32, "num_key_value_heads": 4, "num_heads": 32,
            "full_attn_idxs": [1], "conv_L_cache": 3, "conv_dim": 2048,
        }
        cfg["hidden_size"] = 2048  # rnn head = 2048/32 = 64
        elems = 32 * 64 * 64 + 2048 * 2  # = 135_168
        assert kv.fixed_state_bytes(cfg) == 2 * elems * 2  # 2 gdn layers
        assert kv.kv_bytes_per_token(cfg) == 1024
        d = kv.describe(cfg)
        assert d["attention_type"] == "hybrid-gdn"


class TestSharedKV:
    def test_num_kv_shared_layers_counted_once(self):
        # 8 full layers, k=2 -> 4 caches: per-token = 4*4096 = 16_384
        cfg = llama_like(L=8, heads=32, kv=8, num_kv_shared_layers=2)
        assert kv.kv_bytes_per_token(cfg) == 4 * 2 * 8 * 128 * 2  # = 16_384
        assert kv.kv_bytes_at(cfg, 100) == 16_384 * 100
        assert kv.describe(cfg)["kv_shared_cross_layer"] is True

    def test_yoco_arch_name(self):
        cfg = llama_like(L=8, heads=32, kv=8,
                         model_type="yoco", architectures=["YocoForCausalLM"])
        assert kv.kv_bytes_per_token(cfg) == 1 * 2 * 8 * 128 * 2  # one shared cache
        assert kv.describe(cfg)["kv_shared_cross_layer"] is True

    def test_no_sharing_by_default(self):
        assert kv.describe(llama_like())["kv_shared_cross_layer"] is False


class TestTextScope:
    def test_picks_text_config(self):
        inner = llama_like(L=4, heads=32, kv=8)
        cfg = {"architectures": ["Gemma3ForConditionalGeneration"],
               "model_type": "gemma3", "vision_config": {"hidden_size": 1152},
               "text_config": inner}
        assert kv.text_scope(cfg) is inner
        assert kv.kv_bytes_per_token(cfg) == 2 * 4 * 8 * 128 * 2

    def test_fallback_chain(self):
        for key in ("llm_config", "language_config", "text_model"):
            inner = llama_like(L=2, heads=8, kv=2)
            assert kv.text_scope({key: inner}) == inner

    def test_flat_config_untouched(self):
        cfg = llama_like(L=4)
        assert kv.text_scope(cfg) is cfg

    def test_vision_dict_without_text_fields_ignored(self):
        # a text_config without num_hidden_layers/num_attention_heads is not a backbone
        cfg = llama_like(L=4)
        cfg["text_config"] = {"projection_dim": 768}
        assert kv.text_scope(cfg) is cfg


class TestErrors:
    def test_unknown_model_type(self):
        with pytest.raises(kv.KVConfigError, match="unknown/unregistered"):
            kv.parse_layers({"model_type": "totally_new_arch", "hidden_size": 8})

    def test_missing_geometry(self):
        with pytest.raises(kv.KVConfigError):
            kv.parse_layers({"model_type": "llama", "num_hidden_layers": 4})

    def test_layer_types_length_mismatch(self):
        cfg = llama_like(L=4, model_type="gemma3_text", sliding_window=512,
                         layer_types=["full_attention"] * 3)
        with pytest.raises(kv.KVConfigError, match="length"):
            kv.parse_layers(cfg)

    def test_mla_missing_lora_rank(self):
        cfg = {"model_type": "deepseek_v3", "hidden_size": 7168,
               "num_hidden_layers": 61, "num_attention_heads": 128,
               "qk_rope_head_dim": 64}
        with pytest.raises(kv.KVConfigError, match="kv_lora_rank"):
            kv.parse_layers(cfg)

    def test_bad_T(self):
        with pytest.raises(kv.KVConfigError):
            kv.kv_bytes_at(llama_like(), -1)


class TestParContext:
    def test_pure_gqa_closed_form(self):
        # 4 layers * 4096 B/token = 16_384; weights = 1e9 params * 2 B = 2e9
        cfg = llama_like(L=4, heads=32, kv=8)
        T = kv.kv_par_context(cfg, total_params=1e9)
        assert T == math.ceil(2e9 / 16384)  # = 122_071
        assert kv.kv_bytes_at(cfg, int(T)) >= 2e9
        assert kv.kv_bytes_at(cfg, int(T) - 1) < 2e9

    def test_window_saturation_marker(self):
        # pure window model: max cache = 8 layers * 1024 B * W=512 = 4_194_304 B,
        # far below parity -> returns the saturation context 512
        cfg = llama_like(L=8, heads=8, kv=1, head_dim=256,
                         use_sliding_window=True, sliding_window=512)
        assert kv.kv_par_context(cfg, total_params=1e9) == 512

    def test_pure_recurrent_inf(self):
        # constant cache never reaches parity
        cfg = {"d_model": 512, "n_layer": 4, "ssm_cfg": {}}
        assert kv.kv_par_context(cfg, total_params=1e9) == math.inf

    def test_hybrid_exact(self):
        # 1 full layer (512 B/token) + constant 100_000 B state:
        # T* = ceil((2e9 - 100_000) / 512)
        cfg = {
            "model_type": "falcon_h1", "hidden_size": 512, "num_hidden_layers": 1,
            "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64,
            "mamba_n_heads": 8, "mamba_d_head": 64, "mamba_d_state": 256,
            "mamba_n_groups": 1, "mamba_d_conv": 4, "mamba_expand": 2,
        }
        const = kv.fixed_state_bytes(cfg)
        T = kv.kv_par_context(cfg, total_params=1e9)
        assert kv.kv_bytes_at(cfg, int(T)) >= 2e9
        assert kv.kv_bytes_at(cfg, int(T) - 1) < 2e9
        assert T == math.ceil((2e9 - const) / 512)

    def test_small_target_reached_at_T1(self):
        cfg = llama_like(L=32)
        assert kv.kv_par_context(cfg, total_params=1.0) == 1


# --------------------------------------------------------------------------- #
# real config fixtures (geometry cross-checked by hand)
# --------------------------------------------------------------------------- #

class TestRealConfigs:
    def test_qwen3_8b_gqa(self):
        # llama formula: 2*L*kvh*hd*2 = 2*36*8*128*2 = 147_456 B/token
        cfg = load("Qwen__Qwen3-8B")
        assert kv.kv_bytes_per_token(cfg) == 2 * 36 * 8 * 128 * 2
        assert kv.describe(cfg)["attention_type"] == "gqa"

    def test_qwen25_flag_off_full(self):
        # sliding_window=131072 but use_sliding_window=false -> full attention
        cfg = load("Qwen__Qwen2.5-7B-Instruct")
        assert kv.kv_bytes_per_token(cfg) == 2 * 28 * 4 * 128 * 2  # = 57_344
        assert kv.describe(cfg)["n_window_layers"] == 0

    def test_gpt_neox_mha(self):
        # MHA: 2*44*64*96*2 = 1_081_344 B/token
        cfg = load("EleutherAI__gpt-neox-20b")
        assert kv.kv_bytes_per_token(cfg) == 2 * 44 * 64 * 96 * 2

    def test_gemma2_mirror_1to1(self):
        # unsloth mirror of gemma-2-9b serializes explicit 1:1 layer_types:
        # 21 full + 21 window(W=4096); kv=8, hd=256 -> full marginal
        # 21*2*8*256*2 = 21_504*... = 2*8*256*2=8192/layer -> 172_032 B/token
        cfg = load("unsloth__gemma-2-9b")
        d = kv.describe(cfg)
        assert (d["n_full_layers"], d["n_window_layers"], d["window_size"]) == (21, 21, 4096)
        assert kv.kv_bytes_per_token(cfg) == 21 * 2 * 8 * 256 * 2  # = 172_032
        # at T=8192 (> W): windows saturate at 4096 tokens each
        assert kv.kv_bytes_at(cfg, 8192) == 21 * 8192 * 8192 + 21 * 8192 * 4096

    def test_gemma3_nested_text_config(self):
        # gemma-3-4b-it is multimodal: text_config has 34 layers, kv=4, hd=256,
        # W=1024, pattern 6 -> full at (i+1)%6==0: i=5,11,17,23,29 (5 layers)
        cfg = load("unsloth__gemma-3-4b-it")
        d = kv.describe(cfg)
        assert (d["n_full_layers"], d["n_window_layers"], d["window_size"]) == (5, 29, 1024)
        assert kv.kv_bytes_per_token(cfg) == 5 * 2 * 4 * 256 * 2  # = 20_480

    def test_gpt_oss_20b(self):
        # real config: explicit layer_types 1:1 sliding/full, W=128, kv=8, hd=64
        cfg = load("openai__gpt-oss-20b")
        d = kv.describe(cfg)
        assert (d["n_full_layers"], d["n_window_layers"]) == (12, 12)
        assert kv.kv_bytes_per_token(cfg) == 12 * 2 * 8 * 64 * 2  # = 24_576
        assert kv.kv_bytes_at(cfg, 131072) == 12 * 2048 * 131072 + 12 * 2048 * 128

    def test_deepseek_v2_lite_all_latent(self):
        # 27 latent layers * (512+64) * 2 = 31_104 B/token
        cfg = load("deepseek-ai__DeepSeek-V2-Lite")
        assert kv.kv_bytes_per_token(cfg) == 27 * 576 * 2
        d = kv.describe(cfg)
        assert d["attention_type"] == "mla" and d["n_latent_layers"] == 27

    def test_kimi_k2_is_deepseek_v3_arch(self):
        # Kimi-K2: 61 latent layers * 576 * 2 = 70_272 B/token
        cfg = load("moonshotai__Kimi-K2-Instruct")
        assert kv.kv_bytes_per_token(cfg) == 61 * 576 * 2
        assert kv.describe(cfg)["attention_type"] == "mla"

    def test_jamba(self):
        # 32 layers: attention at i%8==4 -> 4 layers (kv 8, hd 128 -> 4096 B/token
        # per layer = 16_384 B/token); 28 mamba layers with
        # d_inner 8192, d_state 16, d_conv 4: ssm 131_072 + conv 24_576 elems
        # = 155_648 -> 311_296 B per layer -> 8_716_288 B fixed state.
        cfg = load("ai21labs__Jamba-v0.1")
        assert kv.kv_bytes_per_token(cfg) == 4 * 2 * 8 * 128 * 2  # = 16_384
        assert kv.fixed_state_bytes(cfg) == 28 * 155_648 * 2      # = 8_716_288

    def test_granite_4_h_micro(self):
        # 40 layers: 4 attention (kv 8, hd = 2048/32 = 64) -> 8_192 B/token;
        # 36 mamba2: d_inner 4096, d_state 128, n_groups 1, d_conv 4:
        # ssm 524_288 + conv (4096+256)*3 = 13_056 -> 537_344 elems
        # -> 1_074_688 B per layer -> 38_688_768 B fixed.
        cfg = load("ibm-granite__granite-4.0-h-micro")
        assert kv.kv_bytes_per_token(cfg) == 4 * 2 * 8 * 64 * 2  # = 8_192
        assert kv.fixed_state_bytes(cfg) == 36 * 537_344 * 2     # = 38_688_768

    def test_falcon_h1(self):
        # 24 layers, each attention(kv 2, hd 128 -> 512 B/token) + mamba2:
        # d_inner 3072, d_state 256, n_groups 1, d_conv 4:
        # ssm 786_432 + conv (3072+512)*3 = 10_752 -> 797_184 elems -> 1_594_368 B
        cfg = load("tiiuae__Falcon-H1-1.5B-Base")
        # attention per layer: 2 * 2 kv * 128 hd * 2 B = 1_024 B/token
        assert kv.kv_bytes_per_token(cfg) == 24 * 1024  # = 24_576
        assert kv.fixed_state_bytes(cfg) == 24 * 797_184 * 2  # = 38_265_984

    def test_nemotron_h_47b(self):
        # pattern letters: M=mamba, -=mlp, *=attention, E=moe.
        # '*' count in the 98-char pattern = 5 attention layers (kv 8, hd 128)
        # -> 5*4096 = 20_480 B/token; mamba: d_inner 16384, d_state 256,
        # n_groups 8, d_conv 4: ssm 4_194_304 + conv (16384+4096)*3 = 61_440
        # -> 4_255_744 elems -> 8_511_488 B per mamba layer.
        cfg = load("nvidia__Nemotron-H-47B-Base-8k")
        d = kv.describe(cfg)
        assert d["n_full_layers"] == 5 and d["n_recurrent_layers"] == 45
        assert kv.kv_bytes_per_token(cfg) == 5 * 2 * 8 * 128 * 2  # = 20_480
        assert kv.fixed_state_bytes(cfg) == 45 * 4_255_744 * 2

    def test_mamba_pure(self):
        # mamba-2.8b: 64 layers, d_inner 5120, d_state 16, d_conv 4:
        # elems = 5120*16 + 5120*3 = 97_280 -> 194_560 B -> 12_451_840 B total
        cfg = load("state-spaces__mamba-2.8b")
        assert kv.kv_bytes_per_token(cfg) == 0.0
        assert kv.fixed_state_bytes(cfg) == 64 * 97_280 * 2  # = 12_451_840

    def test_recurrentgemma_mirror(self):
        # gated google repo: fixture is the RichardErkhov 4-bit mirror of
        # google/recurrentgemma-2b-it (same config.json + quantization_config).
        # 26 blocks, block_types pattern of 3 -> 8 attention + 18 recurrent;
        # attention kv=1 (MQA), hd 256, window 2048; recurrent = lru 2560 +
        # conv 2560*3 = 10_240 elems -> 20_480 B per layer.
        cfg = load("RichardErkhov__google_-_recurrentgemma-2b-it-4bits")
        d = kv.describe(cfg)
        assert (d["n_window_layers"], d["n_recurrent_layers"]) == (8, 18)
        assert kv.fixed_state_bytes(cfg) == 18 * 20_480  # = 368_640
        assert kv.kv_bytes_at(cfg, 1_000_000) == 8 * 2 * 1 * 256 * 2 * 2048 + 18 * 20_480

    def test_lfm2(self):
        # 16 layers, full_attn_idxs [2,5,8,10,12,14] -> 6 attention (kv 8, hd 64
        # = 2048/32 -> 2*8*64*2 = 2048 B/token per layer); 10 gdn layers:
        # rnn 32*64*64 = 131_072 + conv 2048*2 = 4_096 elems -> 270_336 B
        cfg = load("LiquidAI__LFM2-1.2B")
        assert kv.kv_bytes_per_token(cfg) == 6 * 2048  # = 12_288
        assert kv.fixed_state_bytes(cfg) == 10 * 135_168 * 2  # = 2_703_360

    def test_kimi_linear(self):
        # 27 layers: full (MLA) at 1-based 4,8,...,27 -> 7 latent layers
        # (576*2 = 1152 B/token each); 20 KDA layers: heads 32, head_dim 128:
        # rnn 32*128*128 = 524_288 + conv 3*4096*3 = 36_864 elems
        # -> 561_152 elems -> 1_122_304 B per layer.
        cfg = load("moonshotai__Kimi-Linear-48B-A3B-Instruct")
        d = kv.describe(cfg)
        assert (d["n_latent_layers"], d["n_recurrent_layers"]) == (7, 20)
        assert kv.kv_bytes_per_token(cfg) == 7 * 576 * 2  # = 8_064
        assert kv.fixed_state_bytes(cfg) == 20 * 561_152 * 2

    def test_qwen3_next(self):
        # 48 layers: full at (i+1)%4==0 -> 12 layers (kv 2, hd 256 -> 2048 B/t);
        # 36 gdn: rnn 32*128*128 = 524_288 + conv (2*16*128 + 32*128)*3 = 24_576
        # elems -> 548_864 elems -> 1_097_728 B per layer.
        cfg = load("Qwen__Qwen3-Next-80B-A3B-Instruct")
        d = kv.describe(cfg)
        assert (d["n_full_layers"], d["n_recurrent_layers"]) == (12, 36)
        assert kv.kv_bytes_per_token(cfg) == 12 * 2048  # = 24_576
        assert kv.fixed_state_bytes(cfg) == 36 * 548_864 * 2  # = 39_518_208

    def test_minimax_text_01(self):
        # 80 layers: 10 softmax-full (kv 8, hd 128 -> 4096 B/token each),
        # 70 lightning: 64*128*128 = 1_048_576 elems -> 2_097_152 B per layer.
        cfg = load("MiniMaxAI__MiniMax-Text-01")
        d = kv.describe(cfg)
        assert (d["n_full_layers"], d["n_recurrent_layers"]) == (10, 70)
        assert kv.kv_bytes_per_token(cfg) == 10 * 2 * 8 * 128 * 2  # = 40_960
        assert kv.fixed_state_bytes(cfg) == 70 * 1_048_576 * 2

    def test_mistral_v03_full(self):
        # sliding_window null -> full attention: 2*32*8*128*2 = 131_072
        cfg = load("mistralai__Mistral-7B-v0.3")
        assert kv.kv_bytes_per_token(cfg) == 2 * 32 * 8 * 128 * 2
        assert kv.describe(cfg)["n_window_layers"] == 0

    def test_phi3_full(self):
        # phi3: sliding_window=2047 is vestigial; MHA 32/32 heads, hd 96:
        # 2*32*32*96*2 = 393_216 B/token
        cfg = load("microsoft__Phi-3-mini-4k-instruct")
        assert kv.kv_bytes_per_token(cfg) == 2 * 32 * 32 * 96 * 2
        assert kv.describe(cfg)["n_window_layers"] == 0

    def test_starcoder2_windowed(self):
        # starcoder2: W=4096 always on -> 0 marginal, all 30 layers windowed
        cfg = load("bigcode__starcoder2-3b")
        d = kv.describe(cfg)
        assert (d["n_window_layers"], d["n_full_layers"]) == (30, 0)
        assert kv.kv_bytes_per_token(cfg) == 0.0

    def test_falcon3_llama_gqa_hd256(self):
        # Falcon3-7B: model_type llama, explicit head_dim 256 (not 3072/12):
        # 2*28*4*256*2 = 114_688
        cfg = load("tiiuae__Falcon3-7B-Instruct")
        assert kv.kv_bytes_per_token(cfg) == 2 * 28 * 4 * 256 * 2

    def test_hunyuan_a13b_standard_gqa(self):
        # kv_lora_rank is explicitly null -> NOT MLA; standard GQA
        # 2*32*8*128*2 = 131_072 B/token
        cfg = load("tencent__Hunyuan-A13B-Instruct")
        assert kv.kv_bytes_per_token(cfg) == 2 * 32 * 8 * 128 * 2
        assert kv.describe(cfg)["attention_type"] == "gqa"

    def test_baichuan2_mha(self):
        # no num_key_value_heads -> MHA: 2*32*32*128*2 = 524_288
        cfg = load("baichuan-inc__Baichuan2-7B-Chat")
        assert kv.kv_bytes_per_token(cfg) == 2 * 32 * 32 * 128 * 2
        assert kv.describe(cfg)["attention_type"] == "mha"
