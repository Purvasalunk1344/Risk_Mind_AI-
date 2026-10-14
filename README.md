# Heart Disease Mortality Risk System

End-to-end pipeline (Python, data from SASHELP.HEART only): cleaning, feature
engineering, modelling, survival analysis, explainability, scorecard, fairness
and MLOps monitoring.

## Rebuild from scratch

Data, models and outputs are gitignored, so every teammate must rebuild them.

1. Run `programs/00_export_raw.sas` first (SAS, VS Code Run button).
   It exports SASHELP.HEART to `data/raw/heart_raw.csv`.
2. Install dependencies:
   `pip install --user -r requirements.txt`
3. Run steps 01 to 09 in order, from the `programs` folder:

```bash
cd programs
python 01_eda.py
python 02_cleaning.py
python 03_feature_engineering.py
python 04_split.py
python 05_modeling.py
python 06_evaluation.py
python 07_survival.py
python 08_explainability.py
python 09_scorecard_fairness.py
```

Expected check: champion model is logistic regression, validation AUC about 0.779.

## Notes

- If `lifelines` fails to install, run:
  `pip install --user setuptools wheel`
  `pip install --user --no-build-isolation autograd-gamma`
  `pip install --user lifelines`
- Always read processed CSVs with
  `keep_default_na=False, na_values=[""]`, otherwise the smoking band "None"
  becomes a missing value.
- The test set is final. Do not tune anything based on it.
- Drift monitoring is a simulation: SASHELP.HEART has no time dimension.

## Folder layout

- `programs/`: numbered scripts and `config.py`
- `programs/macros/`: metrics, logging, PSI, scoring helpers
- `data/`, `models/`, `monitoring/`, `output/`: generated (gitignored)
- `docs/model_card.md`: model documentation
