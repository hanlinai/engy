# Qwen3.6-35B-A3B-FP8 — trusted model commitment

The weight-commitment (Merkle) spec the weightless validator pins to verify that
served tokens came from the **unmodified** [`Qwen/Qwen3.6-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
checkpoint. Deterministically derived from the public weights — nothing secret.

## Roots

```
model_root  15ef6d8cf34edcb340a24d0d124303bbe91f0c435cb0041db220bfcc7087c610
lm_root     99ecaeb596fc74700c322ab972f8cfd41f5b40a9804f98c0c18e7dba7ab25ec0
```

`config` (hashed into `model_root`):
`{"arch":"qwen3_5_moe_v0","n_layers":40,"hidden":2048,"d_ff":512,"n_experts":256,"top_k":8,"vocab":248320}`

**Auditable scope:** 40 MoE layers × 256 routed experts + 1 shared expert each +
the router gate + lm_head (10,240 routed-expert roots). Every layer is MoE
(no `first_k_dense_replace`). The token-mixers — Gated DeltaNet (`linear_attn`),
full attention — the MTP head, and the norms live in the trust region; see
`SOUNDNESS.md` (opens with the source).

## What it proves

Every request's commitment binds to this `model_root`; the sampled audit
(`engy.protocol.qwen.verify_qwen_bundle`, dispatched by the validator on
`arch="qwen..."`) checks the served tokens used **these exact weights**, with the
top-k routing the committed router logits support and honest expert GEMMs. The
block-FP8 leaf binds the checkpoint's own `weight` + `weight_scale_inv` bytes
(one f32 scale per 128×128 tile); dequant happens validator-side at recompute.

## Reproduce / verify

Anyone can recompute this from the public checkpoint and confirm the root matches
— that *is* the point (it proves we serve the real model, unmodified):

```bash
python -m engy.miner.qwen_commit --checkpoint <path-to-Qwen3.6-35B-A3B-FP8> \
  --out /tmp/qwen_spec --layers 40 --experts 256
# -> MODEL ROOT: 15ef6d8cf34edcb340a24d0d124303bbe91f0c435cb0041db220bfcc7087c610
```

`model_spec.json.gz` is the full spec (40 layer roots + 10,240 expert roots + the
shared/router roots) — `gunzip` to use. Built CPU-only in ~6 s (16 workers); the
root is reproducible (it matches the milestone-2 value recorded 2026-06-11).
