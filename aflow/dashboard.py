"""Launch the AFlow Streamlit dashboard.

Usage:
    aflow-dashboard                        # looks for workspace/ in cwd
    aflow-dashboard /path/to/project       # looks for workspace/ under given path
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).resolve().parent.parent / "log_viz" / "app.py"

    if not app_path.exists():
        sys.exit(f"Dashboard app not found at {app_path}")

    # First positional arg = project root containing workspace/
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        workspace_root = str(Path(sys.argv[1]).resolve())
        extra_args = sys.argv[2:]
    else:
        workspace_root = str(Path.cwd())
        extra_args = sys.argv[1:]

    env = os.environ.copy()
    env["AFLOW_WORKSPACE"] = workspace_root

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if extra_args:
        cmd.append("--")
        cmd.extend(extra_args)

    raise SystemExit(subprocess.call(cmd, env=env))
