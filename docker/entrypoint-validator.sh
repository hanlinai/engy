#!/usr/bin/env bash
# engy sn53 light validator — entrypoint.
#
# Requires (via --env-file .env.validator):
#   ENGY_SN53_WALLET / ENGY_SN53_WALLET_HOTKEY  the on-chain wallet for set_weights
#
# ENGY_SN53_API and ENGY_SN53_MASTER_HOTKEY are protocol constants with defaults
# in load_config(); they are optional here and must NOT be required, or a
# correctly-configured operator following .env.validator.example never reaches
# Python.

set -euo pipefail

log() { echo "$(date -u '+%Y-%m-%dT%H:%M:%S%z') [entrypoint] $*"; }

log "engy sn53 light validator"

# Fail fast with a clear message rather than a Python traceback. Only the wallet
# names are checked: they are the operator's own and have no default (load_config
# checks them too, this is friendlier).
: "${ENGY_SN53_WALLET:?ENGY_SN53_WALLET is required (your bittensor wallet name)}"
: "${ENGY_SN53_WALLET_HOTKEY:?ENGY_SN53_WALLET_HOTKEY is required (your hotkey name)}"

log "api=${ENGY_SN53_API:-https://provider.engy.ai (default)} netuid=${ENGY_SN53_NETUID:-53} network=${ENGY_SN53_NETWORK:-finney}"
log "wallet=${ENGY_SN53_WALLET}/${ENGY_SN53_WALLET_HOTKEY}"
log "starting poll loop..."

exec engy-sn53-validator "$@"
