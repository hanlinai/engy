#!/usr/bin/env bash
# engy sn53 light validator — entrypoint.
#
# Requires (via --env-file .env.validator):
#   ENGY_SN53_API             engy.web base URL (e.g. https://engy.ai)
#   ENGY_SN53_MASTER_HOTKEY   the pinned master validator hotkey
#   ENGY_SN53_WALLET / ENGY_SN53_WALLET_HOTKEY  the on-chain wallet for set_weights

set -euo pipefail

log() { echo "$(date -u '+%Y-%m-%dT%H:%M:%S%z') [entrypoint] $*"; }

log "engy sn53 light validator"

# Fail fast with a clear message rather than a Python traceback if the two
# required knobs are missing (load_config also checks, this is friendlier).
: "${ENGY_SN53_API:?ENGY_SN53_API is required (engy.web base URL)}"
: "${ENGY_SN53_MASTER_HOTKEY:?ENGY_SN53_MASTER_HOTKEY is required (pinned master hotkey)}"

log "api=${ENGY_SN53_API} netuid=${ENGY_SN53_NETUID:-53} network=${ENGY_SN53_NETWORK:-finney}"
log "starting poll loop..."

exec engy-sn53-validator "$@"
