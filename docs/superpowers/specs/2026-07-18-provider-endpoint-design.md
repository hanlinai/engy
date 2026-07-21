# Light validator: point at engy-provider, surface skipped hotkeys

Date: 2026-07-18

> **Amendment 2026-07-21 — the endpoint below was renamed.**
> `GET /api/subnet/v1/weights/latest` is now
> `GET /api/subnet/v1/epoch/latest`. The old path was **deleted with no
> alias**: no validator was running on chain yet, so the incompatible change
> was free. The old name read as "the vector to submit on chain", which
> undersells the payload — the top-level `weights` is an untrusted
> convenience copy and the authoritative one lives inside the signed
> `result_json` — and the endpoint now sits with the rest of the `/epoch/*`
> family alongside `current` (in flight) and `{index}` (any epoch).
> The payload contract, the signing message and every verification rule in
> this document are unchanged; only the path moved. Implemented in
> `validator/sync.py` `fetch_weights()`. Everything below is the 2026-07-18
> record and still uses the old path.

## Background

The light validator polls `GET {ENGY_SN53_API}/api/subnet/v1/weights/latest`
and verifies the master-signed payload before submitting weights on chain.
The weights service has moved to the engy-provider deployment at
`https://provider.engy.ai`. engy-provider serves the exact same route and
payload contract (`v`, `netuid`, `epoch_index`, `digest`, `result_json`,
`signature`) with the identical signing message
`engy-sn53:epoch:v1:<netuid>:<epoch_index>:<digest>`, so no request or
verification code changes.

uid↔hotkey matching is already enforced: `resolve_uids` (validator/chain.py)
syncs the metagraph at submit time and maps payload hotkeys to uids via
`metagraph.hotkeys`; a hotkey not registered on chain resolves to no uid and
is never submitted. A uid whose hotkey has changed (dereg/re-reg) cannot
inherit the old hotkey's weight because the uid is derived from the hotkey
lookup, never taken from the payload.

## Changes

1. **Config/docs point at the provider.** `.env.validator.example`,
   `README.md`, and `docker/entrypoint-validator.sh` reference
   `https://provider.engy.ai` as the API base URL and describe it as the
   engy-provider service. Only the production URL appears — this is a public
   repo; internal/test environments are never mentioned.
2. **Log skipped hotkeys.** `submit` prints the payload hotkeys that were
   dropped because they are not registered in the metagraph, so an operator
   can see exactly which entries were excluded from set_weights and why.
   The existing behavior (skip silently, refuse to submit when nothing
   matches) is unchanged.

## Out of scope

- No payload-format or verification changes (contract verified identical on
  both sides).
- No new environment variables; `ENGY_SN53_API` keeps its role, only the
  documented value changes.
