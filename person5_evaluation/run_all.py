"""Run the current P5 evaluation from the five-person project structure."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "current/run_all.py"
    subprocess.run([sys.executable, "-B", "-u", str(script)], check=True)
