# Recording and replaying Douglas cassettes

Douglas can operate with a real LLM provider, a deterministic mock, or a replay provider that reuses previously captured outputs. Recording cassettes saves the normalised prompt/output pairs so future runs can execute offline while exercising the full orchestration pipeline.

## When to record

Record cassettes whenever you need a high-fidelity replay of a real model response—typically before running the offline CI pipeline, when preparing demos, or after substantial prompt/configuration changes. The replay provider requires matching cassette entries; if a key is missing Douglas raises a descriptive error that lists the available keys and suggests re-recording.

## One-time recording workflow

1. Ensure networking is allowed (unset the offline guard):
   ```bash
   unset DOUGLAS_OFFLINE
   ```
2. Run Douglas in `real` mode with cassette recording enabled. Pin a seed for reproducibility:
   ```bash
   douglas run --ai-mode real --record-cassettes --seed 123 \
     --cassette-dir .douglas/cassettes
   ```
3. Inspect `.douglas/cassettes/*.jsonl`. Each line contains the metadata, normalised key, and verbatim output captured from the provider.
4. Commit the cassette files alongside your code changes so replay jobs can consume them.

## Switching to replay mode

Replay does not require network access. You can keep `DOUGLAS_OFFLINE` set to `1` for safety, but it is not required because the replay provider never opens a socket.

```bash
douglas run --ai-mode replay --cassette-dir .douglas/cassettes --seed 123
```

The seed, agent label, step name, provider name, and prompt hash must match the recorded cassette for a replay to succeed.

## Repository replay fixture

Douglas keeps a committed cassette set under `tests/replay-fixture/` so CI/CD pipelines can validate replay mode without touching the main project workspace. When you need to refresh those cassettes:

1. Change into the fixture directory so relative paths resolve correctly:
   ```bash
   cd tests/replay-fixture
   ```
2. Allow live provider calls and record a fresh run with the Codex provider:
   ```bash
   unset DOUGLAS_OFFLINE
   douglas run --ai-mode real --provider codex --model gpt-5-codex \
     --record-cassettes --seed 123 --cassette-dir .douglas/cassettes
   ```
3. Replay locally or in CI using the captured JSONL files:
   ```bash
   DOUGLAS_OFFLINE=1 douglas run --ai-mode replay \
     --cassette-dir .douglas/cassettes --seed 123
   ```
4. Commit any updated files under `tests/replay-fixture/.douglas/cassettes/` alongside related changes.

## Refreshing cassettes

Whenever prompts, configuration, or templates change you may need to refresh the cassette set:

1. Delete the outdated JSONL files (or move them to an archive).
2. Re-run the "one-time recording" workflow above with the updated configuration.
3. Commit the refreshed cassettes.

Because keys include a project fingerprint derived from `douglas.yaml` and key path layout, edits that change the project's shape will naturally produce new cassettes.

## Troubleshooting missing-key errors

If Douglas raises a `Replay cassette not found` error:

- Double-check that the `--cassette-dir` and `--seed` arguments match the recording run.
- Inspect the error message. It lists the requested key as well as a sample of available keys from the cassette directory.
- Run the real provider with `--record-cassettes` enabled to populate the missing key.
- Confirm `.douglas/mock.on` is not present if you expected replay mode—this sentinel forces mock mode regardless of CLI options.

For more detail on offline providers and configuration options, see the [README offline modes section](../README.md#offline-modes).
