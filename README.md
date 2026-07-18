# engy

**Verified inference for frontier open models.** Bittensor subnet 53.

engy serves frontier open models on consumer GPUs, with cryptographic
verification that the model you pay for is the model that ran. Each model's
weights and quantization are pinned by a published Merkle root — see
[specs/](specs/) for the roots and how to recompute them from the public
checkpoints. Miners commit to the model's internal activations, and auditors
challenge a random sample: the miner opens the commitment and the opening is
recomputed against the pinned weights. A failed opening is proof of cheating,
not bad luck. No TEEs, no trusted hardware. The proof pins the math, not the
machine.

- API and pricing: [engy.ai](https://engy.ai)
- Subnet: netuid 53 on Bittensor
- Company: [hanlin.ai](https://hanlin.ai)
- Contact: ning@engy.ai

Source (protocol, verifier, miner client, incentive mechanism) opens here as
we approach open miner enrollment.

## Run a light validator

The light validator syncs the master-signed epoch result from the engy
provider API,
verifies the signature against the pinned master hotkey, and submits the same
weight vector on chain. CPU-only; no GPU, no database.

### With Docker (recommended — auto-updating)

The container tracks a GHCR image and [Watchtower](https://containrrr.dev/watchtower/)
pulls new releases automatically, so a running validator stays current without
manual intervention. `last_applied` lives in a named volume, so an update
never re-submits an already-applied epoch.

    cp .env.validator.example .env.validator   # fill in the two required values
    docker compose --env-file .env.validator -f docker/docker-compose.validator.yml up -d
    docker compose -f docker/docker-compose.validator.yml logs -f validator

The wallet is mounted read-only from `~/.bittensor/wallets`. To build and run
from local source instead of the published image, use
`docker/docker-compose.validator-dev.yml` (`up --build`).

Releases: pushing a `v*` tag publishes `ghcr.io/hanlinai/engy:latest` (what
production tracks); pushing to `main` publishes `:staging` for soak-testing via
`docker/docker-compose.validator-staging.yml`.

### Without Docker

    pip install -e .[chain]
    export ENGY_SN53_API=https://provider.engy.ai
    export ENGY_SN53_MASTER_HOTKEY=<published master hotkey>
    export ENGY_SN53_WALLET=<your wallet> ENGY_SN53_WALLET_HOTKEY=<your hotkey>
    engy-sn53-validator

Every payload is verified before it touches the chain: the validator
recomputes `sha256(result_json)` and checks it equals the payload's `digest`
(binding the signature to the exact bytes served, not just a label), verifies
the sr25519 signature over
`engy-sn53:epoch:v1:<netuid>:<epoch_index>:<digest>` against that recomputed
digest, and takes the weight vector from the verified `result_json` — never
from the top-level `weights` field, which is display metadata only. Netuid
match and epoch freshness are also checked (only the last completed epoch is
accepted — a replayed or stale payload is ignored and the previous weights
stay in place). The raw per-miner aggregates behind every digest are public,
so any operator can recompute a closed epoch and falsify a bad result.

## Dev setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pytest tests/ -v
```
