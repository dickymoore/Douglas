import yaml
import subprocess
import sys
from pathlib import Path

from douglas.providers.llm_provider import LLMProvider
from douglas.pipelines import lint, typecheck, test as testpipe

class Douglas:
    def __init__(self, config_path='douglas.yaml'):
        self.config = self.load_config(config_path)
        self.project_name = self.config.get('project', {}).get('name', '')
        self.lm_provider = self.create_llm_provider()

    def load_config(self, path):
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config {path}: {e}")
            sys.exit(1)

    def create_llm_provider(self):
        provider_name = self.config.get('ai', {}).get('provider', 'openai')
        if provider_name == 'openai':
            return LLMProvider.create_provider('openai')
        else:
            print(f"LLM provider '{provider_name}' not supported.")
            sys.exit(1)

    def run_loop(self):
        print("Starting Douglas AI development loop...")
        steps = self.config.get('loop', {}).get('steps', [])
        for step in steps:
            if step == 'generate':
                print("Running generate step...")
                # Placeholder: Invoke LLM generation (not implemented in first commit)
                _ = self.lm_provider.generate_code("""# TODO: describe the task here""")
                continue
            elif step == 'lint':
                print("Running lint step...")
                lint.run_lint()
            elif step == 'typecheck':
                print("Running typecheck step...")
                typecheck.run_typecheck()
            elif step == 'test':
                print("Running test step...")
                testpipe.run_tests()
            else:
                print(f"Unknown step '{step}'; skipping.")
        print("Douglas loop completed.")
        # In later versions, commit/push/PR steps would follow.

    def check(self):
        print("Checking Douglas configuration and environment...")
        print(f"Configuration loaded. Project name: {self.project_name}")

    def doctor(self):
        print("Diagnosing environment...")
        try:
            subprocess.run(['python', '--version'], check=True)
            subprocess.run(['git', '--version'], check=True)
        except subprocess.CalledProcessError:
            print("Error: Required tools are missing.")
        print("Douglas doctor complete.")

    def init_project(self, project_name: str, non_interactive: bool = False):
        print(f"Initializing new project '{project_name}' with Douglas...")
        project_dir = Path(project_name)
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / 'douglas.yaml').write_text('project:\n  name: "' + project_name + '"\n', encoding='utf-8')
        (project_dir / 'README.md').write_text(f"# {project_name}\nCreated with Douglas\n", encoding='utf-8')
        print("Project initialized. Please review and customize douglas.yaml.")
