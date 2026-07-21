# Running an engy light validator

SN53 serves frontier open models on consumer GPUs with cryptographic proof that
the model you paid for is the model that ran. Miners serve inference; an audit
box re-verifies their proofs; the provider scores each epoch and signs the
result.

**Your job is the last step: put that signed result on chain.** You verify it
came from the master hotkey, then `set_weights` on netuid 53. CPU-only — no GPU,
no model weights, no database, no inbound ports.

> The **audit box** (`engyval`) is a different component that also gets called a
> validator. It needs a GPU and is not what this document covers.

## What you need

- A Linux host with Docker and outbound network. CPU-only, no inbound ports.
- A bittensor wallet at `~/.bittensor/wallets`, **unencrypted**, whose hotkey is
  **registered on netuid 53 with a validator permit**.

## Start it

```bash
git clone https://github.com/hanlinai/engy && cd engy
cp .env.validator.example .env.validator   # fill in your two wallet names
docker compose --env-file .env.validator -f docker/docker-compose.validator.yml up -d
docker compose -f docker/docker-compose.validator.yml logs -f validator
```

That is the whole operation. It brings up the validator and a Watchtower
container that keeps it updated; state persists in the `engy_sn53_data` volume
so restarts and updates resume rather than start over.

Startup echoes the wallet it will sign with — check it before walking away:

```
[entrypoint] wallet=<name>/<hotkey>
```

To run from local source instead of the published image, use
`docker/docker-compose.validator-dev.yml` with `up --build`.

---

## Configure

Your whole `.env.validator` is two lines — your wallet and hotkey **names**, as
shown by `btcli wallet list`:

```bash
ENGY_SN53_WALLET=my-wallet         # btcli --wallet.name
ENGY_SN53_WALLET_HOTKEY=my-hotkey  # btcli --wallet.hotkey
```

Everything else — provider URL, master hotkey, netuid, network, intervals, file
paths — already has the right value in code. Leave those lines commented out.

The one case for touching them: pointing at a non-production deployment, which
means setting `ENGY_SN53_API` and `ENGY_SN53_MASTER_HOTKEY` **together**. A
mismatched pair rejects every payload. The full list with defaults is in
`.env.validator.example`.

---

## What to expect once it is running

**It submits every ~24 minutes**, not once per epoch — the chain drops a
validator's weights from consensus if it goes quiet too long, so the same vector
is resubmitted all epoch:

```
[chain] set_weights(42 uids) -> True
[sync] epoch 105: submitted 42 uids (121 blocks since last submit)
```

**In between, every tick says so.** Four ticks in five legitimately skip, and
they report why rather than going quiet:

```
[tick] epoch 105 already submitted; next resubmit in ~11 min (not contacting the chain yet)
```

So a log that has said nothing for a whole poll interval is itself the symptom.

**Fetch failures and rejected payloads are not an emergency.** Nothing
unverified ever goes on chain, and refusing to submit is the designed response —
your last weights stand meanwhile.

Health, if you want to check it:

```bash
docker inspect -f '{{.State.Health.Status}}' engy_sn53_validator
```

Logs only prove the extrinsic was accepted. After a few epochs, confirm with
`btcli` that your weights and `last_update` are current and vtrust is non-zero —
clean submissions with zero vtrust usually means a missing permit or stake.

For the network side, **[provider.engy.ai](https://provider.engy.ai)** is public:
`/epochs` shows each epoch's status and the signed result you are submitting,
`/miners` who is serving and their scores. It is the quickest way to tell a
provider-side problem from one of yours — if an epoch has not finalized there,
your validator having nothing new to submit is correct behaviour.

Updates are automatic: Watchtower pulls new releases and recreates the
container, which is safe mid-epoch. To manage that yourself, pin a release tag
in the compose file and drop the watchtower service.

---

## Troubleshooting

| Log line | What it means |
|---|---|
| `[sync] fetch failed: …` | Provider unreachable. Self-heals. |
| `[sync] payload rejected: …` | The provider served something that failed verification. Escalate — not fixable locally. |
| `[chain] open failed (…)` | Cannot reach the chain. Check `ENGY_SN53_NETWORK` and outbound network. |
| `[chain] set_weights failed (…)` | Rejected at the extrinsic: registration, permit, stake, or rate limit. |
| `[chain] none of the payload's hotkeys are registered on chain` | Nothing to submit; last weights stand. Report it. |
| `[chain] burn goes to X, expected owner Y` | Provider-side misconfiguration. Report it; keep running. |
| `[health] no completed tick in …` | It wedged and restarted itself. Repeatedly means something upstream hangs — usually a chain endpoint that accepts connections and never answers. |

If state is corrupt, stop the container, remove the `engy_sn53_data` volume, and
start it. Do not edit the state file by hand.

---

## Registering on netuid 53

A standard Bittensor operation, not handled by engy's code — consult upstream
documentation and the subnet owner for cost, stake, and permit thresholds.

Nothing checks it locally: an unregistered or permitless hotkey runs fine right
up to the extrinsic, then fails with `[chain] set_weights failed`.

---

## See also

- [SN53 one-pager](SN53_ONE_PAGER.md) — what the subnet is and how it works
- [provider.engy.ai](https://provider.engy.ai) — live epochs, miners, and scores
- [`../README.md`](../README.md) — the repository this validator ships from
