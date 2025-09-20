import subprocess

class GitHub:
    @staticmethod
    def create_pull_request(title: str, body: str, base: str='main', head: str=None):
        if not head:
            head = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], text=True).strip()
        try:
            subprocess.run([
                'gh', 'pr', 'create',
                '--title', title,
                '--body', body,
                '--base', base,
                '--head', head
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to create PR: {e}")
