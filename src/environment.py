"""Environment setup and initialization for workshop."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def ensure_repository_available(
    repo_url: str,
    repo_dir: str,
) -> None:
    """Clone or update repository if src/ directory doesn't exist.

    Args:
        repo_url: URL of the git repository to clone.
        repo_dir: Local directory name for the repository.
    """
    if not Path("src").exists():
        if not Path(repo_dir).exists():
            subprocess.check_call(["git", "clone", repo_url, "-q"])
        else:
            subprocess.check_call(["git", "-C", repo_dir, "pull", "-q"])
        os.chdir(repo_dir)


def ensure_dependencies_installed() -> None:
    """Install required packages if pandas or xgboost are missing."""
    if importlib.util.find_spec("pandas") is None or importlib.util.find_spec("xgboost") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])


def ensure_repo_in_path() -> None:
    """Add repository root to sys.path if not already present."""
    repo_root = Path.cwd()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def setup_environment(
    repo_url: str = "https://github.com/RichardPovinelli/SGA_workshop_deploy_test.git",
    repo_dir: str = "SGA_workshop_deploy_test",
) -> None:
    """Set up the workshop environment.

    Ensures repository is available, dependencies are installed,
    and repo root is in sys.path.

    Args:
        repo_url: URL of the git repository (defaults to workshop deploy repo).
        repo_dir: Local directory name for the repository.
    """
    ensure_repository_available(repo_url, repo_dir)
    ensure_dependencies_installed()
    ensure_repo_in_path()
