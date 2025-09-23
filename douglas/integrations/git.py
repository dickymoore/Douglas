import subprocess
import sys


class Git:
    @staticmethod
    def commit_all(message: str):
        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Git commit failed: {e}")
            sys.exit(1)
