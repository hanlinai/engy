# engy

**Verified inference for frontier open models.** Bittensor subnet 53.

engy serves frontier open models on consumer GPUs, with cryptographic
verification that the model you pay for is the model that ran. Each model's
weights and quantization are pinned by a published Merkle root (see
[specs/](specs/) for the roots and how to recompute them from the public
checkpoints). Every response carries a compact activation fingerprint (TOPLOC),
the check enforced today. A sampled recompute audit, in which an auditor opens
the miner's commitment and re-derives the challenged matrix multiplies against
the pinned weights, is rolling out on top: a failed opening is proof of
cheating, not bad luck. No TEEs, no trusted hardware. The proof pins the math,
not the machine.

**How it works:** the [SN53 one-pager](docs/SN53_ONE_PAGER.md) covers the
players, the proof, scoring and emissions, and gateway routing on one page.

- API and pricing: [engy.ai](https://engy.ai)
- Subnet: netuid 53 on Bittensor
- Company: [hanlin.ai](https://hanlin.ai)
- Contact: ning@engy.ai

Launching in stages: the one-pager, model specs, and the light validator below
are public now; the audit verifier, miner client, and full incentive mechanism
follow.

## Run a light validator

**Full runbook: [docs/VALIDATOR.md](docs/VALIDATOR.md)** — setup, configuration,
what to expect, and troubleshooting. The summary below is the short version.

The light validator syncs the master-signed epoch result from the engy
provider API,
verifies the signature against the pinned master hotkey, and submits that
weight vector on chain, resubmitting it roughly every 120 blocks for the rest
of the epoch. The chain treats a validator that has not submitted within
`activity_cutoff` as inactive and drops its weights from consensus, so a
once-per-epoch submission would leave the validator earning nothing for most
of the week. CPU-only; no GPU, no database.

### With Docker (recommended, auto-updating)

The container tracks a GHCR image and [Watchtower](https://containrrr.dev/watchtower/)
pulls new releases automatically, so a running validator stays current without
manual intervention. Submission state lives in a named volume, so a restart
resumes the resubmit schedule instead of starting the epoch over.

    cp .env.validator.example .env.validator   # fill in your wallet names
    docker compose --env-file .env.validator -f docker/docker-compose.validator.yml up -d
    docker compose -f docker/docker-compose.validator.yml logs -f validator

The poll loop stamps a heartbeat every cycle, whatever the outcome. A container
that stops ticking for three poll intervals shows `unhealthy` in `docker ps`
and exits, so the restart policy recovers it — check it with
`docker inspect -f '{{.State.Health.Status}}' engy_sn53_validator`. Note that
fetch failures and rejected payloads are *healthy*: the spec's failure posture
is to do nothing and let the chain hold the last submitted weights.

The wallet is mounted read-only from `~/.bittensor/wallets`. To build and run
from local source instead of the published image, use
`docker/docker-compose.validator-dev.yml` (`up --build`).

Releases: pushing a `v*` tag publishes `ghcr.io/hanlinai/engy:latest` (what
production tracks); pushing to `main` publishes `:staging` for soak-testing via
`docker/docker-compose.validator-staging.yml`.

### Without Docker

    pip install -e .[chain]
    export ENGY_SN53_WALLET=<your wallet> ENGY_SN53_WALLET_HOTKEY=<your hotkey>
    engy-sn53-validator

Your wallet is the only thing you have to supply. The provider URL and the
master hotkey are protocol constants with built-in defaults — override
`ENGY_SN53_API` and `ENGY_SN53_MASTER_HOTKEY` together only to point at a
non-production deployment.

Every payload is verified before it touches the chain: the validator
recomputes `sha256(result_json)` and checks it equals the payload's `digest`
(binding the signature to the exact bytes served, not just a label), verifies
the sr25519 signature over
`engy-sn53:epoch:v1:<netuid>:<epoch_index>:<digest>` against that recomputed
digest, and takes the weight vector from the verified `result_json`, never
from the top-level `weights` field, which is display metadata only. Netuid
and weight well-formedness are checked too. The raw per-miner aggregates
behind every digest are public, so any operator can recompute a closed epoch
and falsify a bad result.

The validator is deliberately not epoch-aware: it never computes which epoch
is current, and pins no `epoch_s` or `genesis_ts` locally. The provider owns
the timeline. A replayed old payload is stopped instead by a monotonic guard —
an epoch older than the one already applied is never submitted — which is a
local state comparison requiring no clock and nothing the provider can
influence.

## Dev setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pytest tests/ -v
```
