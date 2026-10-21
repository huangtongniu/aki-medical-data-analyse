"""Build the P5 execution notebook from the documented evaluation entry points."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    cells = []

    def markdown(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
                      "execution_count": None, "outputs": []})

    markdown("""# Person 5 — Primary and Stage ≥2 evaluation

This notebook evaluates the fourteen models delivered by Persons 3 and 4 using Person 2's frozen patient split. It runs validation reproduction, separate outcome-specific thresholds, paired patient-stratified bootstrap, sensitivity analyses, comparison tables and figures.

Use the existing **sml** kernel and keep the project folder structure described in `README.md`. Run All uses the companion P5 Python modules. This is the P5 evaluation notebook, not the complete Part 1–5 submission notebook. It does not retrain the models.

Primary model and cutoff preservation is checked against the archived P5 run. Stage ≥2 model development followed the completed primary evaluation; the new validation freeze precedes the new secondary model evaluation. The original files remain archived unchanged.

Contributions: P1 cohort and labels; P2 phenotype QA, split and preprocessing; P3 baseline models; P4 XGBoost models and corrected sparse inference; P5 unified evaluation and verification.
""")
    code("""from pathlib import Path
import os
import sys

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"
sys.dont_write_bytecode = True

working = Path.cwd().resolve()
candidates = [working, working / "current", working / "person5_evaluation/current"]
P5_DIR = next((folder for folder in candidates if (folder / "evaluate.py").is_file()), None)
assert P5_DIR is not None, "Open this notebook in the project, Person 5, or current directory."
assert Path(sys.prefix).name.lower() == "sml", "Select the existing sml kernel."
sys.path.insert(0, str(P5_DIR))

from common import environment
print(environment())
""")
    markdown("""## 1. Statistical and input checks

Check weighted metrics against explicit patient replication and scikit-learn, calibration against an independent GLM, threshold ties, secondary target selection, stratified patient sampling, and explicit structured zeros/NaNs.
""")
    code("""import runpy

try:
    runpy.run_path(str(P5_DIR / "test_evaluation.py"), run_name="__main__")
except SystemExit as result:
    if result.code not in (None, 0):
        raise
""")
    markdown("""## 2. Validation reproduction and threshold freeze

For each model and outcome, use the highest validation probability cutoff attaining sensitivity ≥0.80, including ties. Preserve the original primary cutoffs. Reproduce all supplied validation probabilities and label/ID alignments before new test inference. An existing freeze is verified rather than replaced.
""")
    code("""from threadpoolctl import threadpool_limits
from evaluate import validate

with threadpool_limits(limits=1):
    validate()
""")
    markdown("""## 3. Held-out evaluation and paired bootstrap

The secondary target uses the same full cohort; Stage 1 is secondary-negative. In each of 1,000 bootstrap iterations, sample patients with replacement within their highest observed stage groups (0, 1, ≥2), keeping all eligible encounters. Use the same patient multiplicities for both outcomes and all models. The original P5 bootstrap was unstratified; this version records the updated stratified method explicitly.

Evaluate P2 subsets S1–S4 with fixed models and cutoffs. S3 retains P2's primary-label definition, so it also removes some Stage 1 secondary negatives. Resampling caches are checked against the frozen protocol and can be resumed.
""")
    code("""from evaluate import test

with threadpool_limits(limits=1):
    test()
""")
    markdown("""## 4. Comparison tables and figures

AP is the primary discrimination metric; AUROC and Brier are supporting metrics. Calibration is assessed without changing predictions. The combined table includes 95% intervals and validation-locked operating-point metrics.
""")
    code("""from make_outputs import main as create_outputs

create_outputs()
""")
    markdown("""## 5. Independent verification

Recalculate discrimination and confusion counts, compare calibration with statsmodels, reconstruct selected stratified patient draws, verify all percentile and paired intervals, and confirm the original handoff and archived P5 files remain unchanged.
""")
    code("""from verify_outputs import main as verify_outputs

with threadpool_limits(limits=1):
    verify_outputs()
""")
    code("""import pandas as pd
from IPython.display import display, Image

comparison = pd.read_csv(P5_DIR / "outputs/tables/combined_model_comparison.csv")
columns = ["target", "model_name", "positives", "prevalence", "average_precision", "auroc",
           "brier_score", "threshold", "sensitivity", "specificity", "ppv", "npv"]
display(comparison[columns].round(4))

for name in ["figure1_primary_secondary_pr_roc", "figure2_primary_secondary_calibration",
             "figure3_paired_model_differences", "figure4_sensitivity_analyses"]:
    display(Image(filename=str(P5_DIR / "outputs/figures" / (name + ".png")), width=1100))
""")
    markdown("""The prediction anchor is **at ED admission**. This is internal held-out evaluation of observed creatinine-defined disease, including cases already apparent at first sampling. Intervals condition on the fixed models, cutoffs and patient-stratum counts; they do not include development uncertainty and are not adjusted for multiple comparisons. Primary and secondary AP values must be interpreted with their different prevalences.

See `results_summary_zh.md` and `outputs/tables/combined_model_comparison.csv` for the complete results.
""")
    for index, cell in enumerate(cells):
        cell["id"] = f"p5-v2-{index:02d}"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python (sml)", "language": "python", "name": "sml"},
        "language_info": {"name": "python", "version": "3.12.11"}}, "cells": cells}
    destination = HERE / "P5_primary_secondary_evaluation.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Built {destination.name}: {sum(c['cell_type'] == 'code' for c in cells)} code cells.")


if __name__ == "__main__":
    main()
