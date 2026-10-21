"""Execute and save every cell of the current P5 notebook using sml."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "current/execute_notebook.py"
    subprocess.run([sys.executable, "-B", "-u", str(script)], check=True)
