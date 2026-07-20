# engy: SN53 in one page

*Verified inference on Bittensor **netuid 53**. Buyers hit one OpenAI-compatible
endpoint; permissionless GPU miners serve the tokens; every response carries a
proof that the named model produced it; validators verify a sample and set
weights on chain.*

---

## The players

| Role | Runs | Job |
|---|---|---|
| **Miner** | GPU, inference engine, proof | Serves a committed model, attaches a proof to every response, posts collateral to register. Dials **out** to the gateway (no inbound port). |
| **Gateway** (`api.engy.ai`) | Entry, router, SLA | The product surface: OpenAI/Anthropic APIs, auth and prepaid billing, capacity-aware load-balancing across miners, SSE streaming, fire-and-forget proof capture. |
| **Validator** | Public code, on-chain weights | Samples requests, verifies the proof, writes verdicts, and sets on-chain weights. A proven cheat is slashed. |
| **Chain** | Bittensor netuid 53 | Weights become emissions under Yuma consensus. |

Serving and trust are **deliberately decoupled**: proof capture is a
fire-and-forget tee off the response path, and verification and scoring run
out-of-band. A slow or dead validator never degrades serving.

---

## The proof of "honest" serving

The canonical checkpoint (architecture, weights, and quantization) is pinned as a
32-byte **`model_root`**. Each quantization is its own root under one
buyer-facing model name.

engy combines two lines of trustless-inference verification:

1. **TOPLOC** [1]: a compact, locality-sensitive **activation fingerprint** the
   miner commits with *every* response, at zero extra GPU cost and zero
   TTFT/throughput overhead (an async tee).
2. **Sampled GEMM recompute** [2,3]: on a sampled request the validator
   re-derives the challenged matrix multiplies on CPU, within floating-point
   tolerance. This is the lightweight commit-and-audit family (nearest applied
   twin: CommitLLM [4]; economic soundness of sampled re-checking per
   Proof-of-Sampling [5]), not the heavyweight zero-knowledge route [6].
   Verifying the non-linear attention interior that recompute leaves out is
   emerging work [7].

**What is proven.** Serving with the canonical model checkpoint under the
promised serving SLA. Using a quantized or distilled model to impersonate the
original is considered dishonest.

**What is not proven.** Serving the canonical model at higher precision.

### What the current version actually enforces: TOPLOC

Of the two lines above, **the version shipping today is TOPLOC-only** (we will
impose sampled GEMM recompute in larger models such as GLM 5.2 and Kimi K3).
Every served request carries an activation fingerprint; a validator holding the
canonical checkpoint re-runs the sampled prompt on its own GPU and compares its
top-k against the miner's commitment. The score is the mean top-k mismatch count,
normalized to [0,1] by `topk` (128). The sampled-GEMM recompute path is built but
is not yet the gate; TOPLOC decides `cheat` verdicts today, by a P99 threshold V.

**Experiment: honest FP8 (4090 + 5090) versus an INT4 cheat.** We ran a live test
against Qwen3.6-35B-A3B, with a dedicated validator node and miners split as
follows:

| Miner | Hardware | Serves | P99-Score | Verdict |
|---|---|---|---|---|
| honest-fp8 | RTX 4090 | canonical FP8 checkpoint | ~0.18 | **pass** |
| honest-fp8 | RTX 5090 | canonical FP8 checkpoint | ~0.23 | **pass** |
| int4-cheat | RTX 4090 | INT4 weights, claiming the FP8 `model_root` | ~0.55 | **caught** |

What it showed:

- **Honest FP8 verifies across both consumer generations.** 4090 and 5090
  numerics differ, but nowhere near enough to fail an honest miner, with zero
  false positives over the whole test. Heterogeneous miner hardware is fine.
- **The INT4 cheat is separable at both levels.** Quantizing to INT4 while
  claiming the FP8 root pushes the score to ~0.55, so it is caught easily.

Net: TOPLOC reliably tells an honest FP8 serve apart from a cheaper INT4 serve
masquerading as it, **without** penalizing honest miners on mixed consumer GPUs,
which is exactly the property a permissionless miner set needs.

---

## Architecture: buyer-first, SLA-first, elastic subnet

```
                  buyers  (OpenAI / Anthropic SDKs)
                           │  one endpoint, prepaid
                           ▼
           ┌──────────────────────────────────┐
           │       Gateway  (api.engy.ai)     │
           │   auth · billing · proof tee     │
           │   round-robin + prefix affinity  │
           └──────────┬───────────────┬───────┘
            routed    │               │    routed
            equally   ▼               ▼    equally
         ┌────────────────────┐  ┌─────────────────────┐
         │  1st-party cluster │  │    subnet miners    │
         │  trusted, always   │  │  permissionless,    │
         │  eligible          │  │  eligible once      │
         │                    │  │  qualified + healthy│
         └────────────────────┘  └─────────┬───────────┘
                                           │  proof attached to
                                           │  every response
                                           ▼
                       ┌────────────────────────────────────┐
                       │     validators + chain  (SN53)     │
                       │  weightless, out-of-band:          │
                       │  verify a sample, then set weights │
                       └────────────────────────────────────┘
```

Three principles drive the topology:

- **Buyer-first.** Buyers see one OpenAI/Anthropic-compatible endpoint with
  prepaid billing and never see the subnet. Miners dial out to the gateway (no
  inbound port), so the buyer-facing surface is a single hardened door.
- **SLA-first.** The gateway hands buyer traffic only to workers the control
  plane has marked eligible. The 1st-party cluster is always eligible; a
  permissionless subnet miner becomes eligible only after it qualifies (≥99% HTTP
  and TOPLOC pass, latency ceilings) and stays SLA-healthy, and the circuit
  breaker drops it within about a minute if it degrades. Every request therefore
  lands on a worker already known to meet SLA.
- **Elastic via subnet.** Among eligible workers, the gateway spreads traffic by
  fair round-robin with prefix affinity (a rendezvous hash that keeps a prompt
  prefix on one miner for cache locality), so 1st-party and qualified subnet
  miners share load equally. Adding permissionless GPUs to the subnet raises total
  throughput; an optional per-miner `priority` can bias toward a tier if an
  operator wants it. The detailed routing rules are in *How the gateway routes
  traffic* below.

---

## Scoring & emission split

Settlement is per **(miner, model)** pair, integer-only fixed-point
(floating-point results are not reproducible across validator implementations, so
any arithmetic change is a consensus change). Epochs are **weekly** in prod (4h on
staging), scored fresh from that epoch's slice of the request log.

**Per-(miner, model) score:**

```
tokens(r) = prompt_tokens + completion_tokens
score     = floor( Σ tokens(r) · score_rate[model] / 1000 )   if every gate passes
          = 0                                                 otherwise
```

- **Counted requests.** HTTP 2xx **and** `cost_micro > 0`. Only traffic somebody
  actually paid for scores; free-tier, internal, and probe traffic cannot mint
  weight. This is the **anti-wash-trading** mechanism.
- **`score_rate[model]`.** A per-model blended µUSD-per-1k-tokens rate **we set**,
  deliberately *not* the buyer-facing price card. It makes models comparable on
  one axis, it is the direct lever for steering capacity toward a model, and it is
  immune to buyer discounting: two miners doing identical work score identically
  even if one served a discounted account.
- **Gates.** Four, each a ratio with a threshold and **its own minimum sample,
  below which it passes**. Evaluated in fixed order; the first failure is recorded
  in `gate_reason`:

  | Gate | Grain | Fails when |
  |---|---|---|
  | Acceptance | (miner, model) | 2xx rate `< 99%` |
  | TTFT p99 | (miner, model) | over the model's `qual_ttft_p99_ms` |
  | TPOT p99 | (miner, model) | over the model's `qual_tpot_p99_ms` |
  | Cheat | miner | `cheat` verdicts `> 1%` of rendered verdicts |

  Thresholds are inherited verbatim from the qualifier's unified 1% standard, not
  re-derived. **`unproven` is counted but never gates**: a verdict must
  affirmatively establish misbehaviour before it costs a miner anything, and an
  unusable proof has honest causes. **Scoring never bans**; a failing miner loses
  the epoch and starts clean in the next one. HTTP 499 (buyer hung up) leaves the
  acceptance denominator; every *other* 4xx is emitted before a miner is picked
  and never entered scoring at all, so miner-attributable failure is exactly
  **502** and **504**.

**Score to on-chain weight (`weight_u16`).** One normalization onto 65535: each
row's share is `score_micro ÷ Σ score_micro`, with the rounding remainder going to
the highest-scoring row. There are no model pools and no pool floor; cross-model
comparability comes from `score_rate` instead, which is the same lever with one
fewer mechanism.

**Burn.** If there is no billed traffic (or every score is 0), the whole 65535
goes to the owner hotkey.

**Emission.** The weight vector applied under Yuma consensus. A fresh key starts
at zero (registration churn is unprofitable); a miner that trips any gate earns
nothing that epoch and recovers the next one.

---

## How the gateway routes traffic across 1st-party and subnet

The goal: **hit our own SLA by admitting only qualified, healthy miners, then
scale machine count by adding permissionless subnet GPUs that serve alongside the
1st-party cluster.** Two mechanisms do this, both re-resolved every heartbeat.

### Lever 1: the routable gate (who is *allowed* to serve)

The gateway routes by a **single DB fact** and does no SLA logic itself:

| Worker | Routable when |
|---|---|
| **1st-party** (`type != subnet`) | **connected** (heartbeat-fresh); never gated |
| **subnet** (`type == subnet` + hotkey) | `workers.status == 'active'` **and** heartbeat-fresh |

`'active'` is set by the control plane (**engy-traffic**) only when a subnet
worker is connected **and qualified** (passed a probe: ≥99% HTTP and ≥99% TOPLOC
pass, optional TTFT/TPOT ceilings) **and SLA-healthy**. The **circuit breaker**
also lives in the control plane: it watches `request_log` for fast 502/504s (≥5
failures **and** ≥50% in ~60s) and quarantines by flipping status off `'active'`.
So a sick subnet miner is pulled from routing within about a minute, while the
untrusted subnet can never dent SLA before it has earned `'active'`.

### Lever 2: fair dispatch across eligible miners

Among the eligible miners for a model, the gateway's dispatcher spreads buyer
traffic **fairly**: a rendezvous hash on the prompt prefix (**affinity**) keeps
repeated same-prefix requests on one miner so its radix/prefix cache stays warm,
and everything else is **round-robin**. 1st-party and qualified subnet miners are
peers here, so an active subnet miner receives the same share as a 1st-party
worker. An optional per-miner `priority` lever can bias dispatch toward a tier
(the dispatcher fills the highest live-priority tier before spilling to a lower
one); at equal priority, the default, it reduces to the fair round-robin and
affinity above.

```
buyer request
      │
      ▼
 eligible miners for this model
 (1st-party always; subnet iff status='active')
      │
      ▼
 any with a free slot?  ──no──▶  429 full  /  404 no_miner   (honest signal)
      │ yes
      ▼
 pick within the top live-priority tier:
 prefix affinity if the prompt repeats, else round-robin  ──▶  serve
```

1st-party and active subnet miners share buyer traffic equally, so we scale
machine count with permissionless GPUs **without** letting an unqualified or flaky
miner touch the SLA path: the admission gate, not a routing hierarchy, is what
protects the SLA. And because trust is off the serve path, a slow validator or
audit backlog never slows a single buyer request.

## References

[1] J. M. Ong et al. *TOPLOC: A Locality Sensitive Hashing Scheme for Trustless
Verifiable Inference.* arXiv:[2501.16007](https://arxiv.org/abs/2501.16007), 2025.
The activation-fingerprint scheme the shipping verifier is built on.

[2] R. Freivalds. *Probabilistic Machines Can Use Less Running Time.* IFIP 1977.
The randomized matrix-product verification primitive underlying sampled GEMM
recompute.

[3] P. Anchuri, M. Campanelli, P. Cesaretti, R. Gennaro, T. Jois, H. Kayman,
T. Ozdemir. *Towards Verifiable AI with Lightweight Cryptographic Proofs of
Inference.* IEEE SaTML 2026, arXiv:[2603.19025](https://arxiv.org/abs/2603.19025).
Merkle-commit the inference trace and open a few entries on randomly sampled paths
from output to input; the academic formalization of sampled recompute, including
trace-separation between functionally dissimilar models.

[4] Lambda Class. *CommitLLM: commit-and-audit for open-weight LLM inference.*
[commitllm.com](https://commitllm.com/). Nearest applied twin: Freivalds on the
matmul shell, weightless CPU verifier (~1.3 ms/tok), dense INT8. engy's delta is
MoE, fp8, and live graphed-serve capture.

[5] Y. Zhang, S. Wang, S. Tan, X. Liu, C. Moallemi, R. A. Popa. *Proof of
Sampling: A Nash Equilibrium-Based Verification Protocol for Decentralized
Systems.* arXiv:[2405.00295](https://arxiv.org/abs/2405.00295). Economic basis for
why re-checking a sampled fraction deters dishonest serving.

[6] H. Sun, J. Li, H. Zhang. *zkLLM: Zero Knowledge Proofs for Large Language
Models.* ACM CCS 2024, arXiv:[2404.16109](https://arxiv.org/abs/2404.16109). The
heavyweight zero-knowledge route engy deliberately avoids (~15 min/query, GPU).

[7] *Communication-Efficient Verifiable Attention for LLM Inference.*
arXiv:[2606.16352](https://arxiv.org/abs/2606.16352). Emerging work on verifying
the non-linear attention interior that sampled GEMM recompute omits.

[8] *Verathos: sampled GEMM-proof inference verification.* Bittensor SN96, MIT,
[github.com/verathos-ai/verathos](https://github.com/verathos-ai/verathos). A
comparable weightless-CPU verifier that checks Fiat-Shamir-sampled proofs by
sum-check over Merkle-committed weights. engy's sampled-recompute implementation
is independent.
