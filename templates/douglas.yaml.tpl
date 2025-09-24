project:
  name: "PROJECT_NAME"
  description: "Project description"
  license: "MIT"
  language: "python"
ai:
  default_provider: "codex"
  providers:
    codex:
      provider: "codex"
      model: "code-davinci-002"
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
    security_checks: per_feature
  Designer:
    design_review: per_sprint
    ux_review: per_feature
  BA:
    requirements_analysis: per_sprint
  Stakeholder:
    check_in: per_sprint
    status_update: on_demand
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
    - name: security
      role: DevOps
      activity: security_checks
    - name: test
      role: Tester
      activity: test_cases
    - name: review
      role: Developer
      activity: code_review
    - name: retro
      role: ScrumMaster
      activity: retrospective
      cadence: per_sprint
    - name: demo
      role: ProductOwner
      activity: sprint_review
      cadence: per_sprint
    - name: commit
      role: Developer
      activity: development
    - name: push
      role: DevOps
      activity: release
    - name: pr
      role: Developer
      activity: code_review
  exit_conditions:
    - "ci_pass"
sprint:
  length_days: 10
push_policy: "per_feature"
demo:
  format: "md"
  include:
    - implemented_features
    - how_to_run
    - test_results
    - limitations
    - next_steps
retro:
  outputs:
    - role_instructions
    - pre_feature_backlog
  backlog_file: "ai-inbox/backlog/pre-features.yaml"
history:
  # Tail length (characters) preserved from CI logs and bug tickets. Increase to keep more context at the cost of larger files.
  max_log_excerpt_length: 4000
paths:
  inbox_dir: "ai-inbox"
  app_src: "src"
  tests: "tests"
  demos_dir: "demos"
  sprint_prefix: "sprint-"
  questions_dir: "user-portal/questions"
  questions_archive_dir: "user-portal/questions-archive"
  user_portal_dir: "user-portal"
  run_state_file: "user-portal/run-state.txt"
agents:
  roles:
    - "developer"
    - "tester"
    - "product_owner"
    - "scrum_master"
    - "designer"
    - "ba"
    - "devops"
run_state:
  allowed:
    - "CONTINUE"
    - "SOFT_STOP"
    - "HARD_STOP"
qna:
  filename_pattern: "sprint-{sprint}-{role}-{id}.md"
