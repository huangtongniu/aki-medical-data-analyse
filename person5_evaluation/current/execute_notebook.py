"""Execute all notebook cells in a fresh sml process with local IPython.

This runner does not require Jupyter's localhost kernel sockets. The saved
notebook remains usable with Run All in an ordinary Jupyter sml kernel.
"""
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    if Path(sys.prefix).name.lower() != "sml":
        raise RuntimeError("Run this script with the existing sml Python.")
    runtime = HERE / "outputs/checks/notebook_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True

    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config
    import nbformat

    config = Config()
    config.HistoryManager.enabled = False
    shell = InteractiveShell.instance(config=config)
    path = HERE / "P5_primary_secondary_evaluation.ipynb"
    notebook = nbformat.read(path, as_version=4)
    completed = 0
    previous_directory = Path.cwd()
    os.chdir(HERE)
    try:
        # Clear prior outputs so a failed run cannot retain later-cell results.
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        nbformat.write(notebook, path)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            print(f"Executing notebook code cell {completed + 1}...", flush=True)
            with capture_output(stdout=True, stderr=True, display=True) as captured:
                result = shell.run_cell(cell.source, store_history=True)
            cell.execution_count = result.execution_count
            for name in ["stdout", "stderr"]:
                content = getattr(captured, name)
                if content:
                    cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=content))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
            error = result.error_before_exec or result.error_in_exec
            if error is not None:
                cell.outputs.append(nbformat.v4.new_output("error", ename=type(error).__name__,
                    evalue=str(error), traceback=traceback.format_exception(type(error), error, error.__traceback__)))
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            if error is not None:
                raise RuntimeError(f"Notebook code cell {completed + 1} failed") from error
            completed += 1
            print(f"Notebook code cell {completed} executed successfully.", flush=True)

        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        counts = [cell.execution_count for cell in code_cells]
        embedded_pngs = sum("image/png" in output.get("data", {})
                            for cell in code_cells for output in cell.outputs)
        if counts != list(range(1, len(code_cells) + 1)) or embedded_pngs != 4:
            raise RuntimeError("Notebook execution counts or embedded figure outputs are incomplete.")
        from common import CHECKS, environment, utc_now, write_json
        write_json(CHECKS / "notebook_execution.json", {"executed_at_utc": utc_now(), "notebook": path.name,
            "all_code_cells_passed": True, "code_cells_executed": completed, "environment": environment(),
            "execution_counts": counts, "embedded_png_figures": embedded_pngs,
            "method": "sequential cell execution in a fresh sml process using IPython InteractiveShell",
            "jupyter_kernel_started": False, "kernel_spec_installed_or_changed": False,
            "reason": "Jupyter kernel startup blocked while creating a localhost socket pair; local IPython requires no kernel sockets."})
        from verify_outputs import write_artifact_manifest
        write_artifact_manifest()
    finally:
        os.chdir(previous_directory)


if __name__ == "__main__":
    main()
