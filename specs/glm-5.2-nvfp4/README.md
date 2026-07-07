# GLM-5.2-NVFP4 — trusted model commitment (the 5090 serve)

The weight-commitment (Merkle) spec the weightless validator pins to verify that
served tokens came from the **unmodified** GLM-5.2 weights as quantized to
**NVFP4** (NVIDIA ModelOpt `modelopt_fp4`) for the RTX 5090 serve
(`serve_fp4.sh`, sglang TP8×PP4). Same model as
[`glm-5.2-fp8`](../glm-5.2-fp8/README.md); only the routed-expert quantization
differs, so the `model_root` is **distinct** — both roots are valid for the engy
model id `glm-5.2`, and the validator dispatches on the spec's `quant` field.

## Roots

```
quant       nvfp4
model_root  00800b459b8fca5cff68f33b78d13fdac06fb87a97108c3e19879ff5c48dbdcb
lm_root     a8533e217c3339c518b9431556bbad4f03329feaf8d2a27ed812271b947cec0e
```

`config` (hashed into `model_root`, **identical to the fp8 spec** — same architecture):
`{"arch":"glm_moe_dsa","n_layers":78,"first_k_dense":3,"hidden":6144,"d_ff":2048,"n_experts":256,"top_k":8,"vocab":154880,"routed_scaling_factor":2.5}`

The `model_root` differs from the fp8 serve's purely because (a) the routed-expert
leaves bind different bytes (NVFP4 tiles, not fp8 tiles) and (b) a distinct
domain-separation tag (`MINEMD_GLM_NVFP4_*`). `lm_root` is identical to the fp8
build — lm_head is the same BF16 weights in both quants (a free cross-check).

## What is NVFP4 here

Only the **routed experts** are NVFP4; everything else stays BF16:

```
experts  layers.{L}.mlp.experts.{E}.gate_proj   NVFP4: .weight U8 [d_ff, hidden/2] (two E2M1/byte,
                                                 low-nibble-first) + .weight_scale F8_E4M3
                                                 [d_ff, hidden/16] (micro-block scale, group 16)
                                                 + .weight_scale_2 F32 (per-tensor global)
shared   layers.{L}.mlp.shared_experts.gate_proj.weight   BF16, row tree (committed, not audited)
router   layers.{L}.mlp.gate.weight                       BF16, row tree
bias     layers.{L}.mlp.gate.e_score_correction_bias      F32, single leaf
lm_head  lm_head.weight                                   BF16, row tree
```

Dequant the validator recomputes: `E2M1_LUT[nibble] · e4m3(weight_scale[blk]) ·
weight_scale_2`. The E4M3 block-scale decode is locked bit-exact against
`torch.float8_e4m3fn` on a real expert tile (`tests/test_nvfp4.py`).

**Auditable scope:** 75 MoE layers (model layers 3–77) × 256 routed experts + the
router/`e_score_correction_bias` + lm_head (19,200 expert roots). Layers 0–2 are
dense MLP; the shared expert is committed but not yet separately audited (the
routed-expert GEMM is the FLOP bulk); attention and the DSA indexer live in the
trust region — see `SOUNDNESS.md` (opens with the source).

## What it proves

Every request's commitment binds to this `model_root`; the sampled audit
(`engy.protocol.glm_nvfp4.verify_glm_nvfp4_bundle`) checks the served tokens used
**these exact NVFP4 weights**, with honest sigmoid + noaux_tc top-k routing and
honest expert GEMMs. Same proof shape as the fp8 path
(`GLM52_PROOF.md`, opens with the source).

## Reproduce / verify

Recompute from the NVFP4 checkpoint and confirm the root matches:

```bash
python -m engy.miner.glm_nvfp4_commit --checkpoint <path-to-GLM-5.2-NVFP4> --out /tmp/glm_nvfp4_spec
# -> MODEL ROOT (NVFP4): 00800b459b8fca5cff68f33b78d13fdac06fb87a97108c3e19879ff5c48dbdcb
```

`model_spec.json.gz` is the full spec (75 layer roots + 19,200 expert roots + the
router-bias table) — `gunzip` to use. Built CPU-only in ~228 s (8 workers).
