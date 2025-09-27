#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
FIXTURE_DIR="$ROOT_DIR/tests/replay-fixture"

cleanup() {
  rm -rf "$FIXTURE_DIR/.douglas/workspaces"
}

pushd "$FIXTURE_DIR" > /dev/null
trap cleanup EXIT

rm -rf "$FIXTURE_DIR/.douglas/workspaces"

unset DOUGLAS_OFFLINE || true

if [ -z "${CODEX_HOME:-}" ]; then
  export CODEX_HOME="$HOME/.codex"
fi
if [ -d "$CODEX_HOME" ] && [ -f "$CODEX_HOME/auth.json" ]; then
  export CODEX_AUTH_FILE="$CODEX_HOME/auth.json"
fi

douglas run --ai-mode real --record-cassettes --seed 123 --cassette-dir .douglas/cassettes "$@"

cleanup

popd > /dev/null
