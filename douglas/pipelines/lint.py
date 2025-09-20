import subprocess
import sys

def run_lint():
    try:
        subprocess.run(['ruff', '.'], check=True)
        subprocess.run(['black', '--check', '.'], check=True)
        subprocess.run(['isort', '--check', '.'], check=True)
        print("Lint checks passed.")
    except subprocess.CalledProcessError:
        print("Lint checks failed.")
        sys.exit(1)
