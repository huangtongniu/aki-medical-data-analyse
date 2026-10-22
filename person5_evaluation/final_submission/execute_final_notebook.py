"""Execute the real notebook sequentially in a fresh sml/IPython process.

This avoids Jupyter localhost sockets. It does not bypass authentication or use
synthetic/cached data. Output is a separate executed notebook and execution log.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def main():
    if Path(sys.prefix).name.lower() != "sml":
        raise RuntimeError("Use the existing sml Python.")
    verification = HERE / "verification"
    verification.mkdir(exist_ok=True)
    os.environ["IPYTHONDIR"] = str(verification / "ipython")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config

    config = Config()
    config.HistoryManager.enabled = False
    shell = InteractiveShell.instance(config=config)
    source = HERE / "Final_AKI_End_to_End.ipynb"
    output = verification / "Final_AKI_End_to_End.executed.ipynb"
    notebook = nbformat.read(source, as_version=4)
    os.chdir(HERE)
    started = datetime.now(timezone.utc).isoformat()
    counts, timings, error_details = [], [], None
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    total = sum(c.cell_type == "code" for c in notebook.cells)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        index = len(counts) + 1
        tag = cell.metadata.get("tags", ["untagged"])[0]
        print(f"Real run: cell {index}/{total} [{tag}]", flush=True)
        start = time.monotonic()
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
        cell.execution_count = result.execution_count
        for name in ("stdout", "stderr"):
            value = getattr(captured, name)
            if value:
                cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=value))
                print(value[-2000:], flush=True)
        for rich in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata))
        error = result.error_before_exec or result.error_in_exec
        timings.append({"cell": index, "tag": tag, "seconds": time.monotonic() - start, "passed": error is None})
        if error is not None:
            cell.outputs.append(nbformat.v4.new_output("error", ename=type(error).__name__, evalue=str(error),
                traceback=traceback.format_exception(type(error), error, error.__traceback__)))
            error_details = {"cell": index, "tag": tag, "type": type(error).__name__, "message": str(error)}
        else:
            counts.append(cell.execution_count)
        nbformat.validate(notebook)
        nbformat.write(notebook, output)
        if error_details:
            break
    passed = error_details is None and len(counts) == total
    report = {"started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "sequential cells in a fresh sml process with IPython InteractiveShell",
        "jupyter_kernel_started": False, "real_source_run": True, "synthetic_or_cached_inputs_used": False,
        "source_notebook_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "all_code_cells_passed": passed, "completed_code_cells": len(counts), "total_code_cells": total,
        "error": error_details, "timings": timings,
        "run_directory": str(shell.user_ns.get("RUN_DIR", "")), "executed_notebook": output.name}
    (verification / "real_run_execution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if error_details and "RUN_DIR" in shell.user_ns:
        status_path = shell.user_ns["RUN_DIR"] / "checks/run_status.json"
        status_path.write_text(json.dumps({"status": "blocked_source_access" if error_details["tag"] == "bigquery_access" else "failed",
            "finished_at_utc": report["finished_at_utc"], "error": error_details,
            "real_data_run_completed": False, "existing_data_or_models_used": False}, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("all_code_cells_passed", "completed_code_cells", "error")}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
