"""Build the notebook from the executable source, optionally execute it."""
from pathlib import Path
import argparse
import sys

import nbformat as nbf


HERE = Path(__file__).resolve().parent
INTRO = """# Person 3 secondary Stage >=2 AKI baselines

**COMP90089 Group 7 · 20 October 2026 · Secondary handoff v1**

Four new models use `aki_stage2_label`: training prevalence, age/gender LR,
structured L2 LR and structured-plus-chief-complaint L2 LR. The full modelling
cohort, frozen patient split and Person 2 fitted preprocessing are retained.
Stage 1 and operational negatives are coded 0. The code cells contain the full
Person 3 workflow; the accompanying preprocessing module is byte-identical to
Person 2's original.

The secondary endpoint was in the methodology, but implementation began after
primary test evaluation. This bounded plan is saved before secondary tuning;
it is not claimed to predate inspection of primary test results. Only train and
validation files are used. Secondary test evaluation remains with Person 5.
Primary models and frozen evaluation records are not changed.
"""
SETUP = """## Environment and file locations

Install the pinned packages in `requirements.txt` into a dedicated environment
and restart the notebook kernel if needed. The core scientific package versions
match Person 2's saved preprocessing. Python 3.12 was used to execute this notebook.

Local example, from `ML For Health`:

```sh
python3 -m venv .venv-person3
.venv-person3/bin/python -m pip install -r "Group Project/person3_stage2_handoff/requirements.txt"
```

For Colab, upload/unzip the private team files first. Keep this layout:

```text
Group Project/
  aki_handoff/predictor_columns.csv
  person2_handoff/person2_split_preprocessing/
    aki_preprocessing.py
    outputs/...
  person3_stage2_handoff/
    aki_preprocessing.py
    P3_stage2_baselines.ipynb
```

Set `PROJECT_DIR` below if the project folder is elsewhere. In Colab, install
packages before importing them, for example:

```python
%pip install -r "/content/Group Project/person3_stage2_handoff/requirements.txt"
```

No BigQuery queries or credentials are needed for these baselines.
"""
PRELUDE = """import sys
from pathlib import Path

# Set to a folder containing person2_handoff, or keep None for automatic discovery.
PROJECT_DIR = None

search_bases = ([Path(PROJECT_DIR).expanduser().resolve()] if PROJECT_DIR is not None
                else [Path.cwd().resolve(), *Path.cwd().resolve().parents])
module_dirs = []
for base in search_bases:
    module_dirs.extend([base, base / "person3_stage2_handoff",
                        base / "Group Project/person3_stage2_handoff"])
module_dir = next((p for p in module_dirs if (p / "aki_preprocessing.py").is_file()), None)
if module_dir is None:
    raise FileNotFoundError("Keep aki_preprocessing.py beside this notebook, or set PROJECT_DIR.")
sys.path.insert(0, str(module_dir))
"""
EXPLANATIONS = {
    "Imports and predefined development choices": """## Predefined modelling choices

The constant model learns the training encounter prevalence. The age/gender
classifier uses only shared standardised age and the male indicator; female is
the reference. Its coefficients are not regularised or tuned. Structured LR
uses all 27 structured features, including eight missingness indicators, with
L2 regularisation. Structured-plus-text LR adds the 2,772 frozen TF-IDF terms to
the same structured columns, for 2,799 features. Each regularised LR's C is
selected independently from the same six fixed values by maximum validation AP;
ties within 1e-12 favour smaller C. No class weighting or resampling is used.
The text model uses lbfgs, tol 1e-8 and max_iter 20000; the age/gender and structured models use max_iter 2000. No XGBoost-specific sparse-zero correction is applied to LR.
""",
    "Load development data and verify the shared handoff": """## Handoff validation

Check the manifest, patient separation, identifier order, predictor whitelist,
package versions, saved matrices and training-only preprocessing statistics.
Test patient assignments in the manifest are checked for separation, but no
test outcomes, predictors or matrices are opened. Reproducing the saved matrices
with the original fitted preprocessors also checks prediction row alignment.
Verify identical structured columns between LR variants and identical TF-IDF
settings, vocabulary and IDF between lr_text and tree_text. A fresh training-only
vectorizer fit is used for QA, not as a replacement model input.
""",
    "Fit the four baselines using bounded validation searches": """## Baseline fitting and tuning functions

Models are fitted only on training data. Validation is used to select the
structured and text models' C values and describe development performance. No train/validation
refit, cross-validation, threshold optimisation or probability recalibration is
performed. Convergence warnings stop execution rather than being ignored.
""",
    "Save pipelines and validation predictions, and test round-trip reproducibility": """## Reproducible pipelines and prediction exports

Each saved sklearn Pipeline accepts the original 13 cleaned predictor columns.
The age/gender classifier receives only its two selected columns. The constant
classifier ignores all feature values. Verify that raw-input, matrix-input and
reloaded-pipeline probabilities agree, and that CSV probabilities round-trip.
Predictions retain stay_id and subject_id for Person 5's paired evaluation.
""",
    "Validation plots and a short methods/results contribution": """## Plotting and report contribution functions

Plot precision–recall, ROC and ten-quantile calibration curves. AP uses
`average_precision_score`, not trapezoidal PR area. Identical predictions in
the constant model form a single calibration point. No confidence bands are
shown; patient-level uncertainty estimation belongs to the shared evaluation.
""",
    "Run the complete development workflow": """## Complete workflow helper

`run(PROJECT_DIR)` is the command-line equivalent of the execution cells below.
Do not call it again after those cells unless deliberately rerunning the workflow.
""",
}


def build(execute=False):
    source = (HERE / "stage2_baselines.py").read_text()
    source = source.split('\nif __name__ == "__main__":')[0]
    cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_markdown_cell(SETUP),
             nbf.v4.new_code_cell(PRELUDE)]
    for block in source.split("# %% ")[1:]:
        title, _, code = block.partition("\n")
        cells.extend([nbf.v4.new_markdown_cell(EXPLANATIONS[title]), nbf.v4.new_code_cell(code.strip())])
    cells.extend([
        nbf.v4.new_markdown_cell("""## Execute checks and save the plan before tuning

Only aggregate results are displayed. The runtime source hashes identify the
exact upstream artifacts used in this run.
"""),
        nbf.v4.new_code_cell("""from IPython.display import display, Image, Markdown

project_root = find_project_dir(PROJECT_DIR)
out = project_root / "person3_stage2_handoff/outputs"
out.mkdir(parents=True, exist_ok=True)
(out / "analysis_plan.json").write_text(json.dumps(development_plan(), indent=2) + "\\n")
data, checks = load_development_data(project_root)
display(pd.DataFrame(checks))
display(pd.DataFrame([
    {"partition": part, "encounters": len(data[part]["y"]),
     "patients": data[part]["labels"].subject_id.nunique(),
     "positives": int(data[part]["y"].sum()),
     "prevalence": data[part]["y"].mean()}
    for part in ["train", "validation"]
]))
"""),
        nbf.v4.new_markdown_cell("## Fit models and inspect the complete bounded tuning record"),
        nbf.v4.new_code_cell("""result = develop_baselines(data)
display(result["tuning"])
print("Selected structured C:", result["selected_C"])
print("Selected structured + text C:", result["selected_text_C"])
display(result["metrics"])
"""),
        nbf.v4.new_markdown_cell("## Save the private handoff and verify pipeline/CSV reloads"),
        nbf.v4.new_code_cell("""out = save_handoff(project_root, data, result, checks)
make_validation_figure(out, data, result)
write_report_contribution(project_root, data, result)
display(pd.DataFrame(checks))
print(f"All {len(checks)} checks passed.")
print("No test outcomes, predictors, matrices or predictions opened/generated.")
"""),
        nbf.v4.new_markdown_cell("## Validation discrimination and calibration"),
        nbf.v4.new_code_cell('display(Image(filename=str(out / "figures/validation_discrimination_calibration.png")))'),
        nbf.v4.new_markdown_cell("## Methods, validation results and interpretation for report integration"),
        nbf.v4.new_code_cell('display(Markdown((project_root / "person3_stage2_handoff/P3_stage2_report_contribution.md").read_text()))'),
        nbf.v4.new_markdown_cell("""## Handoff status and next steps

Four training-fitted Stage >=2 pipelines and aligned validation predictions are
available in this separate folder. `target` and each pipeline's `target_name_`
identify `aki_stage2_label`; model names must be combined with the target when
integrating primary and secondary outputs. Validation performance is used for
selection and is not a final test estimate. No threshold or confidence interval
is calculated here.

Person 5 should adapt the primary-only evaluator for the secondary endpoint,
freeze the secondary metric/threshold/resampling protocol, and coordinate final
evaluation with Person 4's new Stage >=2 models. Do not reuse primary thresholds
or change frozen primary records. The XGBoost representation correction is not
needed for LR; the shared text vocabulary and IDF remain unchanged.

Files: `outputs/validation_metrics.csv`, `outputs/tuning_record.csv`,
`outputs/validation_predictions.csv`, `outputs/models/*.joblib`,
`outputs/handoff_checks.csv`, `outputs/model_metadata.json`,
`P3_stage2_report_contribution.md`, `DECISIONS_STAGE2.md`,
`P5_stage2_integration.md` and `README_STAGE2.md`.
"""),
    ])
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.14"},
    })
    if execute:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        km = KernelManager(kernel_name="python3")
        km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        class ReportingNotebookClient(NotebookClient):
            def process_message(self, msg, cell, cell_index):
                if msg["msg_type"] == "stream":
                    print(msg["content"]["text"], end="", flush=True)
                return super().process_message(msg, cell, cell_index)
        client = ReportingNotebookClient(nb, timeout=1200, km=km, resources={"metadata": {"path": str(HERE)}})
        client.execute()
    nbf.validate(nb)
    output = HERE / "P3_stage2_baselines.ipynb"
    nbf.write(nb, output)
    print(f"Notebook saved: {output.name}; cells: {len(nb.cells)}; executed: {execute}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    build(parser.parse_args().execute)
