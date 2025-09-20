import subprocess
import sys

def run_typecheck():
    try:
        subprocess.run(['mypy', '.'], check=True)
        print("Type check passed.")
    except subprocess.CalledProcessError:
        print("Type check failed.")
        sys.exit(1)
