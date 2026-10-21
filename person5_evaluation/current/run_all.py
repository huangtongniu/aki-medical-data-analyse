"""Run the complete Person 5 update in the current Python environment."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    steps = [["test_evaluation.py"], ["evaluate.py", "validate"], ["evaluate.py", "test"],
             ["make_outputs.py"], ["verify_outputs.py"]]
    for step in steps:
        print("Running: " + " ".join(step), flush=True)
        subprocess.run([sys.executable, "-B", "-u", *step], cwd=HERE, check=True)


if __name__ == "__main__":
    main()
