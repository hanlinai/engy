# GLM-5.2-FP8 — trusted model commitment

The weight-commitment (Merkle) spec the weightless validator pins to verify that
served tokens came from the **unmodified** [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8)
checkpoint. Deterministically derived from the public weights — nothing secret.

## Roots

```
model_root  a4cb49fa0e4809ca71af337fe86ce52ec04e0fcb00a3ed01637f5c75eb2ee8e9
lm_root     a8533e217c3339c518b9431556bbad4f03329feaf8d2a27ed812271b947cec0e
```

`config` (hashed into `model_root`):
`{"arch":"glm_moe_dsa","n_layers":78,"first_k_dense":3,"hidden":6144,"d_ff":2048,"n_experts":256,"top_k":8,"vocab":154880,"routed_scaling_factor":2.5}`

**Auditable scope:** 75 MoE layers (model layers 3–77) × 256 routed experts + 1
shared expert each + the router/`e_score_correction_bias` + lm_head (19,200 expert
roots total). Layers 0–2 are dense MLP; attention and the DSA indexer live in the
trust region — see `SOUNDNESS.md` (opens with the source).

## What it proves

Every request's commitment binds to this `model_root`; the sampled audit
(`engy.protocol.glm.verify_glm_bundle`) checks the served tokens used **these exact
weights**, with honest sigmoid + noaux_tc top-k routing and honest expert GEMMs. See
`GLM52_PROOF.md` (opens with the source).

## Reproduce / verify

Anyone can recompute this from the public checkpoint and confirm the root matches —
that *is* the point (it proves we serve the real model, unmodified):

```bash
python -m engy.miner.glm_commit --checkpoint <path-to-GLM-5.2-FP8> --out /tmp/glm_spec
# -> MODEL ROOT: a4cb49fa0e4809ca71af337fe86ce52ec04e0fcb00a3ed01637f5c75eb2ee8e9
```

`model_spec.json.gz` is the full spec (75 layer roots + 19,200 expert roots + the
router-bias table) — `gunzip` to use. Built CPU-only in ~93 s (8 workers).
