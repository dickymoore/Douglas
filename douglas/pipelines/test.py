import subprocess
import sys

def run_tests():
    try:
        subprocess.run(['pytest', '-q'], check=True)
        print("Tests passed.")
    except subprocess.CalledProcessError:
        print("Tests failed.")
        sys.exit(1)
