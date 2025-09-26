#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
FIXTURE_DIR="$ROOT_DIR/tests/replay-fixture"

pushd "$FIXTURE_DIR" > /dev/null
trap 'rm -rf .douglas/workspaces' EXIT

rm -rf .douglas/workspaces

unset DOUGLAS_OFFLINE || true

douglas run --ai-mode real --record-cassettes --seed 123 --cassette-dir .douglas/cassettes "$@"

popd > /dev/null
