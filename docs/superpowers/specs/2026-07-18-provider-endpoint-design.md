# Light validator: point at engy-provider, surface skipped hotkeys

Date: 2026-07-18

## Background

The light validator polls `GET {ENGY_SN53_API}/api/subnet/v1/weights/latest`
and verifies the master-signed payload before submitting weights on chain.
The weights service has moved to the engy-provider deployment at
`https://provider.engy.ai`. engy-provider serves the exact same route and
payload contract (`v`, `netuid`, `epoch_index`, `digest`, `result_json`,
`signature`) with the identical signing message
`engy-sn53:epoch:v1:<netuid>:<epoch_index>:<digest>`, so no request or
verification code changes.

uid↔hotkey matching is already enforced: `resolve_uids` (engy_sn53/chain.py)
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
