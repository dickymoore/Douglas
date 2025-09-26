#!/bin/sh
set -euo pipefail

unset DOUGLAS_OFFLINE || true
export DOUGLAS_OFFLINE=1

douglas run --ai-mode replay --cassette-dir ./.douglas/cassettes --seed 123
