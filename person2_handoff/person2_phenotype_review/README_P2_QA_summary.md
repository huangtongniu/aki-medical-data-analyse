# Person 2 — Phenotype review and descriptive analysis

*Prepared 13 October 2026. Inputs: Person 1's `aki_handoff/` export and `ML_for_health_Group_Project.ipynb`.*

## Verdict

**Person 1's extraction is accepted. No discrepancies were found that change any count, label or predictor value.**

- All 30 count-reconciliation checks pass. Every number in the README and in the methodology document is reproduced from the exported files.
- All 41 row-level rule checks on `full_cohort_audit.csv` give 0 violations. These cover the window, sampling fields, outcome states, labels, stages and timing order.
- An independent re-implementation of the predictor cleaning reproduces `modelling_data.csv` exactly: 0 disagreeing rows across all 13 predictors.
- 26 of 26 synthetic boundary tests of the outcome logic pass.

The source creatinine tables stay inside the authorised BigQuery environment, so the outcome code was checked in two ways instead of re-running it on source rows:
- line-by-line code review;
- synthetic patients run through a re-expression of notebook cells 27–64.

## What was checked (task 1–2)

| Area | Result |
|---|---|
| 24 h / discharge endpoint | `min(arrival + 24 h, discharge)`. Samples are included at both boundaries. A sample 1 min after 24 h, or after an earlier discharge, is excluded. |
| Strictly earlier comparators | References exactly 48 h and exactly 7 d earlier are included; 48 h + 1 min and 7 d + 1 min are excluded. A same-time record is never its own comparator. The **minimum** reference is used. |
| Decimal thresholds | Exact. For example, 2.3 − 2.0, 4.0 − 3.7 and 0.7 − 0.4 all fail `>= 0.3` in float but pass in Decimal. 1.65 vs 1.5 × 1.1 is the same. |
| Conflicting samples | All records at a conflicting patient/time are removed, both as current samples and as references. |
| Overlapping encounters | A shared sample goes to the most recent arrival; on an exact tie, the smaller `stay_id` wins. |
| Admission matching | Labs from another admission are not current samples but can serve as references. Labs with a missing `hadm_id` count as current. |
| Operational negative | ≥ 2 current times + timed comparison + no rise. |
| Stages | Ratio ≥ 2 gives stage 2. Ratio ≥ 3, or ≥ 4.0 mg/dL with a qualifying rise, gives stage 3. 4.0 → 4.1 is not a rise. |
| Items | 50912 and 52546 count. Whole-blood 52024 is ignored. Zero values are screened out. |

### Observations (not errors; for the team to note)

1. **Asymmetric ascertainment.** 384 positives (4.9%) have a single current sample, judged against an earlier, often pre-arrival, reference. The other 9,114 single-sample encounters with an assessable comparison and *no* rise are indeterminate, not negative. This follows from the agreed definition. All 384 are in the "already positive at first sample" group. It is tested by sensitivity subset S4.
2. **Redundant condition.** With ≥ 2 distinct in-window times, the later sample always has a comparator. "Usable timed comparison" is therefore always satisfied (0 exceptions among 112,713).
3. **Stage 3 absolute route** accepts a qualifying rise by either criterion (48 h +0.3 or 7 d ×1.5). This is consistent with the written definition.
4. **CSV trap.** With default `pd.read_csv`, 24 raw pain entries (`"n/a"` × 21, `"NA"` × 3) in `labelled_predictors_before_cleaning.csv` become NaN. Anyone re-cleaning from the raw files must use `keep_default_na=False`. `modelling_data.csv` is **not** affected.
5. **Cosmetic issues** in the extraction notebook: the cell 58 heading says "stage >2" (the code correctly uses ≥ 2); there are some typos; and the printed label in cell 40 has an encoding artefact.

## Descriptive results (task 3–6)

### Positives vs operational negatives (`table1`, `table2`)
Denominators: 7,866 positive and 105,231 negative encounters.

- **Missingness is informative.** Vital signs are missing about twice as often in positives:
  - temperature: 19.3% vs 9.9%;
  - all six vitals missing: 15.2% vs 7.3%, |SMD| ≈ 0.25–0.27.
  - Among encounters missing all six vitals, the AKI rate is 13.5%, compared with 6.4% when vitals are observed. **Person 3 should seriously consider missingness indicators** rather than median imputation alone.
- Positives more often arrive by ambulance (57.4% vs 50.5%) and less often walk in (33.1% vs 44.4%).
- Positives more often have acuity 1 (19.2% vs 12.0%) and are more often male (55.9% vs 49.6%).
- Positives have lower blood pressure: median SBP 127 vs 132 mmHg, DBP 71 vs 74 mmHg.
- Median age is the same in both groups (66).

### Labelled vs excluded encounters (`table3`, `table3b`)
Denominators: 113,097 labelled, 72,171 indeterminate and 10,072 no-usable-creatinine encounters.

**There is strong selection.** Compared with the labelled population, excluded encounters are:
- younger: median age 55 (indeterminate) and 52 (no creatinine) vs 66;
- discharged sooner: median hospital stay 1.1 and 0.4 days vs 3.7 days.

The no-creatinine group mostly has short windows: 74% were discharged within 24 h.

Excluded encounters also have *less* vital-sign missingness. The model is therefore developed on older, sicker patients who stay longer and have repeat creatinine tests. Results do not generalise to all admitted ED patients. This belongs in the report's limitations.

### Ascertainment (`table4`)
Positives are sampled more densely than negatives:
- current samples, median: 3 vs 2;
- span ≥ 12 h: 81% vs 69%.

Negatives' last sample is a median of 7.5 h before the end of the window. A negative therefore means "no rise among the samples taken", not "monitored throughout".

### First-sample groups among the 7,866 positives (`table5`)

| Group | n (%) | Hours to first qualifying sample, median | Stage ≥ 2 |
|---|---|---|---|
| Already positive at first sample | 2,627 (33.4%) | 0.8 | 31.7% |
| Assessable, no rise, later positive | 725 (9.2%) | 15.6 | 20.3% |
| Unassessable at first, later positive | 4,514 (57.4%) | 14.8 | 28.2% |

A third of positives already met the definition at the first sample, about 0.8 h after arrival, so they were probably present at arrival. The report should frame the task as *early identification*, not new-onset prediction. This matches the stated aim.

## Proposed sensitivity subsets (task 7; to agree with Person 5)

These are defined only from extraction and ascertainment fields, which are **never model inputs**. Row-level flags are in `sensitivity_subset_flags.csv` (`subject_id`, `stay_id` and booleans only).

| ID | Subset | Encounters | Positives | Prevalence |
|---|---|---|---|---|
| — | All labelled | 113,097 | 7,866 | 6.96% |
| **S1** | Exclude all-six-vitals-missing | 104,274 | 6,673 | 6.40% |
| S2 | Complete cases | 89,902 | 5,510 | 6.13% |
| **S3** | Exclude positives already positive at first sample | 110,470 | 5,239 | 4.74% |
| **S4** | ≥ 2 current samples for positives too (symmetric) | 112,713 | 7,482 | 6.64% |
| S5 | Negatives only if sampling span ≥ 12 h | 80,751 | 7,866 | 9.74% |
| S6 | Full 24 h follow-up only | 106,710 | 7,501 | 7.03% |

**Recommended minimum set: S1, S3 and S4.** Each should be run as *evaluation of the frozen model in the subgroup*, with patient-to-partition assignment unchanged. The subsets change prevalence, so subgroup AP must be read against subgroup prevalence.

## Notes for Person 3 (split, preprocessing, baselines)

- **Keep patients together.** 66,090 patients contribute 113,097 encounters (up to 69 per patient). 3,788 patients have both positive and negative encounters, so a patient-grouped split is essential.
- Missingness is outcome-related (see above), so add missingness indicators, at least for vitals, acuity and pain. `pain_status` already separates "missing" from "other recorded".
- Chief complaint has inconsistent casing ("Abd pain" / "ABD PAIN"), so lowercase it before TF-IDF.
- Acuity 4–5 is rare (251 + 8). Consider treating acuity as ordinal numeric or merging 4–5.
- 59 encounters have DBP ≥ SBP, and extreme values pass the agreed broad screens (HR 1, SBP 1, SpO₂ 2%). No change is needed, but tree models and scaled LR handle these differently.
- Do **not** use any column from `sensitivity_subset_flags.csv` or `full_cohort_audit.csv` as a predictor.

## Data handling and small cells
- The row-level outputs (`sensitivity_subset_flags.csv`) and the handoff data must stay in approved private storage. Only aggregate tables go into the shared document.
- Some aggregate cells hold fewer than 10 encounters: acuity 5 (n = 8) and acuity 4 within stage ≥ 2 (n = 4). Merge or suppress these before publishing, per the MIMIC data use agreement.

## Files

| File | Content |
|---|---|
| `P2_phenotype_review.ipynb` | Executed notebook. Reruns locally (`../aki_handoff`) or on Colab (`/content/aki_handoff`). |
| `outputs/P2_tables.xlsx` | All aggregate tables, one sheet each |
| `outputs/qa_count_reconciliation.csv` | 30 count checks: expected vs observed |
| `outputs/qa_rule_checks.csv` | 41 row-level rule checks (violations) |
| `outputs/qa_cleaning_reproduction.csv` | Independent cleaning vs `modelling_data.csv` |
| `outputs/qa_synthetic_boundary_tests.csv` | 26 synthetic boundary cases |
| `outputs/table1_characteristics_pos_vs_neg.csv` | Characteristics table with SMDs |
| `outputs/table2_missingness_by_outcome.csv` | Missingness by outcome |
| `outputs/table3_labelled_vs_excluded.csv` (+ `table3b_…`) | Selection comparison and its largest imbalances |
| `outputs/table4_ascertainment_by_state.csv` | Sampling and ascertainment by outcome state |
| `outputs/table5_first_sample_groups_positives.csv` | First-sample groups among positives |
| `outputs/sensitivity_subset_counts.csv` | Subset sizes and prevalence |
| `outputs/sensitivity_subset_flags.csv` | Row-level subset flags (**private**) |

CSV files are UTF-8 with BOM, so they open correctly in Excel.
