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

Four principles drive the topology:

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
- **Zero data retention.** The gateway keeps no buyer content: prompts and
  completions are relayed and never persisted (only metadata such as token
  counts and status is logged), and even the verification tee (the TOPLOC proof,
  which necessarily carries buyer tokens) lives only in expiring RAM and never
  touches disk. Our **1st-party clusters run under the same zero-retention
  policy**, so a request served on the 1st-party path is **ZDR-guaranteed end to
  end** by construction.

---

## Scoring & emission split

Settlement is integer-only fixed-point (floating-point results are not
reproducible across validator implementations, so any arithmetic change is a
consensus change). Epochs are **weekly** in prod (1h on staging), scored fresh
from that epoch's data.

Each epoch's 65535 weight budget is split in two, each normalized on its own:

| Share | Budget | Who earns it | Paid for |
|---|---|---|---|
| **80%, baseline pool** | 52428 | enrolled servers in the current **hardware classes** | being online and serving whatever model we assign |
| **20%, traffic** | 13107 | every bound, unbanned miner | paid buyer traffic that passes the four gates |

### Baseline pool (80%): the mechanism

The baseline pool pays for **capacity we want to keep**, independent of how
much buyer traffic the router happened to send a given machine. It is defined by
a set of **hardware classes that we review and adjust periodically** as model
demand and the hardware market change. Which classes exist, how many servers of
each we want and what each is worth are configuration; the mechanism below is
what stays fixed.

- **Hardware classes.** A class is a server shape (e.g. a standard 8-GPU
  server of one GPU type) plus the minimum it must meet for the models we plan
  for it, such as model-disk size. Only servers that match a current class are
  enrolled, and every qualifying server counts as **one unit** of its class. The
  hardware is read by engy itself during onboarding (GPU class, count and
  memory, GPU IDs not claimed by another server, model-disk size), not taken
  from the server's own report, and it is re-checked on every sample.
- **Sub-pools.** The pool is split into one sub-pool per class, in proportion
  to **how many servers of that class we want × its base price**, the base
  price reflecting what one such server costs to rent. Each sub-pool is divided
  among its own servers only, so one class never dilutes another, and it is
  **always paid out in full**: fewer servers than the target means a larger
  share each. An empty sub-pool burns its share. The split is frozen into each
  epoch's signed params, so an adjustment applies from the next epoch on.
- **Counting starts when it really serves.** Enrolment alone earns nothing.
  A server starts counting at its first sample in which a worker has deployed
  a model, connected to the gateway, passed verification and is active; from
  then on every sample counts, up or down.
- **Model-agnostic.** Enrolled servers agree that we may switch the model they
  serve at any time, as demand requires. Which model a server runs never
  affects its share. While **we** switch a server's model, it counts as online
  for a grace window (1h); a switch the owner starts earns no grace.
- **Uptime, sampled.** Every 5 minutes each counting server is sampled once. A
  server is **up** in a sample if **any** of its workers is connected and
  active. A sample slot where the sampler itself was down is dropped for
  everyone, so a sampler outage costs nobody.
- **Continuous uptime (vesting).** Inference must be highly available, so the
  pool pays for **continuous** service. A server's uptime is held as pending
  until it has been in the pool for **5 days without an outage** (10 minutes or
  more down). It is then paid in full in that epoch, back pay included, and paid
  as it goes from then on. An outage before that forfeits the pending part and
  restarts the clock. Spot/preemptible capacity that is reclaimed after a few
  days therefore earns nothing from the pool. A single missed sample is not an
  outage; slots where many servers are down at once (an outage on our side) and
  model switches we start never count against a server.
- **Share within a sub-pool.** `pool_score = paid_slots × 1e6 ÷ observed_slots`,
  where `paid_slots` are the vested up slots paid in the epoch (back pay
  included) and `observed_slots` the epoch's sampled slots; the sub-pool is
  divided by `pool_score`. A sub-pool whose servers are all still pending burns
  its share. A server that joins or leaves mid-epoch earns only the slots it
  was counting for.
- **Public.** Each epoch's pool rows (`server_id`, class, uptime, outages,
  vesting state, weight) are in the signed result and on the epoch, miner and
  leaderboard pages.

### Baseline pool: current configuration

For the current period the classes are **B300 and B200, split 70 / 30**, with a
**5-day** vesting period. Both apply from **epoch 23**:

| Class | Server | Model disk | Base price | Share of pool | Share of all emission |
|---|---|---|---|---|---|
| B300 | standard 8x B300 | ≥ 2 TB | 200 | **70%** | 56% |
| B200 | standard 8x B200 | ≥ 1 TB | 100 | **30%** | 24% |

B300 is the larger class because we want more capacity for the largest models;
B200 stays as the cost-effective server for the high-volume flash models. When
the classes are next adjusted, this table changes; the mechanism above does not.

### Traffic (20%)

The traffic budget uses the same per-(miner, model) score as before the split,
unchanged, and is open to every miner, enrolled or not.

**Traffic score, per (miner, model):**

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

**Score to weight (`weight_u16`).** Each budget is normalized separately: a
row's share of its budget is `score ÷ Σ score`, with the rounding remainder
going to that budget's top row. A miner's on-chain weight is the sum of its
pool and traffic rows. Cross-model comparability inside the traffic budget comes
from `score_rate`.

**Burn.** A budget with nothing to pay (no enrolled server online, or no
scoring traffic) goes to the owner hotkey; the other budget is unaffected.

**Emission.** The weight vector applied under Yuma consensus. A fresh key starts
at zero (registration churn is unprofitable); a miner that trips a traffic gate
earns no traffic share that epoch and recovers the next one.

**Activation.** The 80/20 split takes effect from a cutover epoch announced in
advance. Until then, 100% of the budget follows the traffic score.

---

## How the gateway routes traffic across 1st-party and subnet

The goal: **hit our own SLA by admitting only proven, healthy miners, then scale
throughput by letting permissionless subnet GPUs serve alongside the 1st-party
cluster.** The gateway stays dumb and fast: it relays tokens and routes on one
fact, whether a worker is cleared to serve. Deciding who is cleared belongs to the
control plane, **engy-traffic**, which earns a new miner its place with **probe
traffic** and **TOPLOC proof** before any buyer request reaches it.

### Onboarding a new subnet miner

**Who may onboard.** A server in a current **baseline-pool hardware class**
(today a standard 8-GPU B300 or B200 server; see the baseline pool) onboards
after basic verification:
SSH reach, its miner key, and its hardware as engy reads it. Any other server must either pass
**TEE attestation** (a verified CPU-TEE quote) or belong to a **whitelisted**
(approved) provider; otherwise onboarding is refused.

A permissionless miner is a stranger, so it proves itself before it touches paid
traffic. engy-traffic drives **synthetic probe traffic** at a newly connected
worker: real requests it serves as if in production, but that no buyer sees and
that can never earn on-chain weight. Two things must hold across that probe:

- **It serves correctly.** High acceptance and latency inside the model's SLA.
- **It serves the real model.** Every probe response carries a **TOPLOC proof**,
  and the validator path confirms the activations came from the canonical
  checkpoint, not a cheaper quant wearing its name.

Clear both and the worker becomes **active**: the one state in which the gateway
hands it real buyer traffic. Fall short and it never leaves the waiting room; the
miner fixes its serve and re-onboards. This admission gate is what keeps an
untrusted subnet from ever denting the SLA.

```
 new permissionless miner
        │
        ▼
   probe traffic          served like production, but no buyer sees it
        │                 and it earns no weight
        ├── serves correctly?  (acceptance + latency inside the SLA)
        └── TOPLOC proof?      (activations match the canonical model)
        │
     both hold
        ▼
     active  ─────▶  serves production traffic, equal peer of 1st-party
        │
   degrades / slows
        ▼
     pulled  ─────▶  must re-prove before it serves again
```

### Staying active

Qualification is not a one-time badge. engy-traffic keeps watching every active
worker and pulls any that starts failing or slows below its SLA, within about a
minute, then makes it re-prove itself before it serves again. Trust is earned
continuously, and it is earned **off the buyer's path**: a slow or backed-up
validator never slows a single production request.

### Sharing the load

Once active, a subnet miner is a **peer of the 1st-party cluster**, not a
second-class fallback. The gateway spreads buyer traffic across all eligible
miners evenly, with prompt-prefix affinity so a conversation stays on a
cache-warm miner. Adding permissionless GPUs simply adds capacity: the same SLA,
more machines serving it. An operator can still bias toward a tier when it wants
to.

Net: **the admission gate, not a routing hierarchy, protects the SLA.** A
permissionless GPU touches paid traffic only after it has served probe traffic
cleanly and proven via TOPLOC that it ran the canonical model, and it is pulled
the moment it degrades. That is how we scale machine count with untrusted GPUs
without ever putting the SLA at risk.

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
