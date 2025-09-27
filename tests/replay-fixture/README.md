# Replay Fixture

This minimal project exists purely to record and replay Douglas cassettes inside CI/CD pipelines. Keep prompts and code stable so the captured responses remain valid. The configuration runs five iterations per recording (`max_iterations`) to give the dashboard richer sample data while leaving the loop deterministic (only the `test` step executes).
