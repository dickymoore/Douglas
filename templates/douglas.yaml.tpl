project:
  name: "PROJECT_NAME"
  description: "Project description"
  license: "MIT"
  language: "python"
ai:
  provider: "openai"
  model: "gpt-4"
  prompt: "system_prompt.md"
loop:
  steps: ["lint", "typecheck", "test"]
