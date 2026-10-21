"""Run from project root or this folder using the activated sml interpreter."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script):
    subprocess.run([sys.executable, "-B", script], cwd=HERE, check=True)


if __name__ == "__main__":
    run("test_evaluation.py")
    if not (HERE / "outputs/checks/test_run_started.json").exists():
        run("run_validation.py")
    run("run_test_evaluation.py")
    run("make_report.py")
    run("verify_outputs.py")
