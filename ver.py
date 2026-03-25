import subprocess
def get_git_version():
    try:
        version = subprocess.check_output(
            ["git", "describe", "--tags", "--dirty", "--always"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        commit_msg = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        return commit_msg
    except Exception:
        return "no git info"
__version_commit__ = get_git_version()
import re
match = re.search(r"^\d+\.\d+\.\d+", __version_commit__)
__version__ = match.group() if match else "0.0.0"

# MAJOR.MINOR.PATCH
