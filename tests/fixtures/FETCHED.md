# Real config.json fixtures (tests/fixtures/raw_configs/)

Fetched from https://huggingface.co/<repo>/resolve/main/config.json on
2026-10-10 with a plain User-Agent (`kv-census/0.1`), one request at a time
(>= 0.35 s apart). Cached verbatim; committed so `make test` is fully
offline. These are TEST fixtures only (census snapshots live under
data/raw/configs/, owned by census-data).

## Original publishers (ungated)
- Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-8B, Qwen/Qwen3-Next-80B-A3B-Instruct
- tiiuae/Falcon3-7B-Instruct, tiiuae/Falcon-H1-1.5B-Base
- bigcode/starcoder2-3b, EleutherAI/gpt-neox-20b
- microsoft/Phi-3-mini-4k-instruct, allenai/OLMo-2-1124-7B-Instruct
- internlm/internlm2-7b, THUDM/glm-4-9b-chat-hf
- moonshotai/Kimi-K2-Instruct, moonshotai/Kimi-Linear-48B-A3B-Instruct
- tencent/Hunyuan-A13B-Instruct (also bullerwins/Hunyuan-A13B-Instruct-hf,
  the HF-native re-release; identical geometry)
- baichuan-inc/Baichuan2-7B-Chat, stabilityai/stablelm-2-1_6b
- mistralai/Mistral-7B-v0.3
- openai/gpt-oss-20b, deepseek-ai/DeepSeek-V2-Lite
- ibm-granite/granite-4.0-h-micro, ai21labs/Jamba-v0.1
- nvidia/Nemotron-H-47B-Base-8k
- state-spaces/mamba-2.8b, state-spaces/mamba2-2.7b (original release format)
- tiiuae/falcon-mamba-7b
- LiquidAI/LFM2-1.2B, MiniMaxAI/MiniMax-Text-01

## Mirrors (original repo is gated; config.json identical except
## quantization_config / transformers_version where applicable)
- unsloth/gemma-2-9b            <- google/gemma-2-9b is gated (401).
  NOTE: this mirror serializes explicit 1:1 alternating layer_types;
  original google configs carry sliding_window_pattern=6 (5:1 local:global).
  kv.py honors whichever the config contains.
- unsloth/gemma-3-4b-it         <- google/gemma-3-4b-it gated
- unsloth/gemma-3-1b-it         <- google/gemma-3-1b-it gated
- RichardErkhov/google_-_recurrentgemma-2b-it-4bits
                                 <- google/recurrentgemma-2b-it gated
  (4-bit quantized mirror; config.json is google's + quantization_config)


## Census wave-2 fixtures (2026-10-10, second batch)
Copied verbatim from the census cache `data/raw/configs/` (fetched by the
census-data agent from the same repos; provenance = the repo id in the
filename). Includes e.g. allenai/DataDecide-c4-10M (hf_olmo), Qwen/Qwen3.5-0.8B,
inclusionAI/Ling-2.5-1T, RWKV/RWKV7-1.5B-20260805, apple/OpenELM-1_1B,
nvidia/Llama-3_1-Nemotron-51B-Instruct, nvidia/NVIDIA-Nemotron-3-Ultra-550B,
ibm-granite/granite-swash-2b + granite-switch-4.1-3b, LGAI-EXAONE 3.5/4.5,
ServiceNow-AI/Apriel-H1, Snowflake/Llama-3.1-SwiftKV-8B, deepseek-ai/DeepSeek-V4-Flash,
EleutherAI/olmo3-7b, allenai/MolmoE-1B, MiniMaxAI M1/M2, internlm/Atria-Dawn-Preview
(glm_moe_dsa), moonshotai/Kimi-VL-A3B (untyped MLA config),
microsoft/Phi-4-mini-flash + bitnet-b1.58, stabilityai japanese-stablelm-alpha,
zai-org/cogvlm-base-224, baichuan-inc/Baichuan-Audio-Base, Qwen VL families,
openbmb MiniCPM3 + MiniCPM-SALA, inclusionAI/Ring-flash-linear-2.0,
tencent/Hy4-preview, tiiuae/falcon-11B, Snowflake/snowflake-arctic,
01-ai/Yi-VL-34B, Qwen/Qwen-14B, internlm/Agent-FLAN-7b, EleutherAI GPT-2 +
bigcode/astraios (gpt_bigcode), bigcode/octogeex (chatglm), microsoft/Phi-3-vision,
HuggingFaceTB/SmolVLM2-2.2B-Base (intentionally underdetermined: used to assert
KVConfigError), LGAI-EXAONE/EXAONE-3.5.
