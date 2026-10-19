
AKI ED DATA HANDOFF
==================

Sources:
MIMIC-IV-ED and MIMIC-IV Hospital v2.2.
One row per ED encounter; patients can have multiple encounters.

Files:
- modelling_data.csv: cleaned predictors, identifiers and two outcomes.
- full_cohort_audit.csv: all eligible encounters, outcome states and
  ascertainment fields. Predictor values precede section 6 cleaning.
- labelled_predictors_before_cleaning.csv: original predictor values
  for the labelled encounters.
- predictor_missingness.csv: missingness after cleaning.
- outcome_counts.csv: counts of the four outcome states.
- predictor_columns.csv: the only fields intended as model inputs.

Population:
195,340 eligible encounters.
113,097 labelled encounters from 66,090 patients.
7,866 primary positives and 105,231 operational negatives.
72,171 indeterminate encounters.
10,072 encounters without usable current creatinine.
2,254 secondary positives in the labelled population.

Outcomes:
aki_label: 1 = observed qualifying creatinine rise; 0 = operational negative.
aki_stage2_label: 1 = observed creatinine stage >=2; 0 = no observed stage >=2
within the primary labelled population. Stage 1 encounters are 0 here.

Follow-up starts at ED arrival and ends at 24 hours or hospital discharge,
whichever occurs first.

Primary positives use the observed 48-hour absolute-rise or seven-day
relative-rise comparisons implemented in the notebook.
This is a creatinine-only endpoint.

Operational negatives require:
- At least two distinct usable current sample times.
- A usable timed comparison.
- No observed qualifying creatinine rise.

The older historical-baseline exclusion was removed by agreement.
This changes the previous 103,772-encounter labelled population.
Operational negatives do not establish that AKI was clinically absent.
Positives include cases identified at the first current sample.

Cleaning:
Temperature: retain values >70 and <120; assume values 30–43 are Celsius
and convert to Fahrenheit.
Heart rate: >0 and <300.
Respiratory rate: >0 and <70.
Oxygen saturation: >0 and <=100.
Systolic pressure: >0 and <400.
Diastolic pressure: >0 and <300.
Acuity: 1–5 inclusive.
These are broad study screens, not normal clinical ranges.
Pain: accept numeric entries 0–10, including trailing decimal zeros.
Other pain entries remain distinguished from missing entries by pain_status.
Chief complaint: trim whitespace and treat empty text as missing.

Next team member:
Use predictor_columns.csv to select inputs explicitly.
Do not use identifiers, outcome fields or ascertainment fields as predictors.
Create shared patient-separated partitions using subject_id.
Fit imputation, encoding, scaling and text processing on training data only.
No imputation, encoding or model fitting has been performed.
This export does not include a split assignment.
CSV files do not preserve pandas data types; check types after loading.

The notebook is the detailed extraction and phenotype specification.
