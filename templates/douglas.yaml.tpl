project:
  name: "PROJECT_NAME"
  description: "Project description"
  license: "MIT"
  language: "python"
ai:
  provider: "openai"
  model: "gpt-4"
  prompt: "system_prompt.md"
cadence:
  Developer:
    development: daily
    quality_checks: daily
    code_review: per_feature
  Tester:
    test_cases: daily
  ProductOwner:
    backlog_refinement: per_sprint
    sprint_review: per_sprint
  ScrumMaster:
    daily_standup: daily
    retrospective: per_sprint
  DevOps:
    release: per_feature
loop:
  steps:
    - name: generate
      role: Developer
      activity: development
    - name: lint
      role: Developer
      activity: quality_checks
    - name: typecheck
      role: Developer
      activity: quality_checks
    - name: test
      role: Tester
      activity: test_cases
    - name: review
      role: Developer
      activity: code_review
    - name: commit
      role: Developer
      activity: development
    - name: push
      role: DevOps
      activity: release
    - name: pr
      role: Developer
      activity: code_review
