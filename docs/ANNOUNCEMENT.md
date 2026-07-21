# engy (SN53): verified inference for frontier open models

[engy](https://engy.ai) is an inference provider. Buyers hit one public endpoint and get frontier
open models; a permissionless, global fleet of GPU miners serves the tokens. Every
response carries a cryptographic commitment that the named model produced it, and
weightless validators verify a sample and set weights on chain.

engy is already a live inference product with paying buyers, not a subnet waiting
for demand to appear. The gateway at `api.engy.ai` speaks both the **OpenAI Chat
Completions** and **Anthropic Messages** APIs, so Cursor, Codex, Claude Code,
Hermes, and OpenClaw connect directly with no adapter. Miners earn against revenue
that already exists, and **only paid work scores**.

## Proven on real hardware

Both models run on our own consumer-GPU clusters today, verified end to end
against the committed roots.

- **GLM-5.2** (753B total / 40B-active DSA MoE), in production as **NVFP4 on our
  RTX 5090 cluster** (PP4×TP8, 32 GPUs across 4 nodes): ~31 tok/s single-stream,
  flat past 140k context, 256k-token window. Validated at **both FP4 and FP8**,
  FP4 quality matching FP8 down to identical greedy outputs. GLM-5.2 ships
  datacenter-only (H100/H200/B200); we were first to run the full 753B on consumer
  cards.
- **Qwen3.6-35B-A3B** (35B / 3B-active MoE, FP8): one 8× RTX 4090-48G node serves
  ~4,500 tok/s at 128 concurrent users (157 tok/s single-stream), scaling past
  8,600 at 1,024 users. A single 4090 runs it.

TOPLOC adds no extra GPU work: the fingerprint is a byproduct of the normal
forward pass, teed off the response path and verified out-of-band, so a slow or
backed-up validator never touches a live buyer request. The capture ships for
**sglang** today (any GPU vendor); other serving stacks need their own hook.

## For miners

**At launch we open one model for mining: Qwen3.6-35B-A3B.** We start with the
smallest launch model on purpose, to keep the entry bar low. It serves on any
FP8-capable NVIDIA GPU (Ada, Hopper, or Blackwell) that fits the model's
~35 GB of weights plus KV cache: **2× RTX 4090** or **2× RTX 5090**, a single
**L40S** or **RTX 6000 Ada** (48 GB), or a datacenter card (**H100, H200, or
B200**). More hardware, more throughput. GLM-5.2 stays 1st-party for now and opens
to miners later.

```
buyer → api.engy.ai → gateway → your miner → your sglang serve → your GPUs
```

What you do:

- **Serve the committed model.** Run Qwen3.6-35B-A3B and match its published
  `model_root` (architecture, weights, quantization). Serving a cheaper quant
  while claiming the original is what gets caught.
- **Attach a TOPLOC proof to every response.** That is what makes the inference
  verifiable. Miner setup and the proof contract are in
  `github.com/hanlinai/engy/blob/main/docs/MINER.md`; the scoring-eligible
  reference miner code is being polished and released soon. The bare
  `engy-refminer` sends an empty commitment and is **not scoring-eligible**.
- **Pass the qualification probe.** A new worker is probed before it receives
  buyer traffic, and stays routed only while it is healthy.

**Scoring** is per (miner, model), per epoch (currently daily):

```
W = floor( Σ (prompt_tokens + completion_tokens) · score_rate[model] / 1000 )
```

times a gate that is **0 or 1**. Four checks, and the first failure zeroes your
epoch: **accept** (2xx rate below 99%), **ttft** and **tpot** (p99 past the
model's target), **cheat** (over 1% of audits return `cheat`). Each passes
automatically below its minimum sample, so a quiet epoch will not fail you on
noise. Gates carry no state: a bad epoch costs that epoch and nothing more. Scores
normalize to 65535. Only requests that returned 2xx **and** actually billed count,
so unpaid traffic is worth zero.

## For validators

Two roles, and **as a third party you only run the light one.**

**Master validator (engy, GPU-backed).** It holds the canonical checkpoints and
does the actual verification: each epoch it re-computes the sampled TOPLOC proofs
on its own GPUs, cross-checks every miner's commitment against the real model,
scores each (miner, model) pair, and publishes one **signed** epoch weight result.
This is the only role that needs GPUs, and during stabilization engy runs it.

**Light validator (anyone, CPU-only).** **No GPU, no model weights.** Each epoch it
pulls the master-signed result from the provider API (no credential, no
registration), audits it (verifies the sr25519 signature against the pinned master
hotkey, resolves hotkeys to uids, renormalizes), and calls `set_weights` on netuid
53. It checks and mirrors the master's weights on chain, nothing heavier. Full
setup: `github.com/hanlinai/engy/blob/main/docs/VALIDATOR.md`.

**What the audits mean.** Weights come from sampled two-phase proof audits: the
nonce is revealed only after the master holds the commitment, so the challenge
cannot be ground. Audits return `pass`, `cheat`, or `unproven`; only `cheat` costs
a miner anything, and `unproven` is treated as a pass, since an unusable proof has
honest causes. Verification is sampled and probabilistic, not a full proof of the
forward pass.

Why the split: one verification implementation while the mechanism settles, and no
day-one GPU requirement to validate. Weight formation is concentrated at the master
for now, a stabilizing measure, not the end state (see The plan).

## The plan

**Phase 1 (now):** production stack, permissionless miner registration on
Qwen3.6-35B, daily epochs on chain, engy as master validator. Expect score rates,
gate targets, and sampling to move as they meet real traffic; we announce changes
before they land.

**Phase 2:** more models open to miners (GLM-5.2 next), and proof verification
distributes. Validators run the audits themselves, the master role recedes,
parameters settle. The trigger is operational, not calendar: stable settlement
across epochs and a verification path reproducible outside our infrastructure.
