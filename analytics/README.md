# Module 2 — Titanic Analytics and Modeling

This module is a single offline-reproducible analytics workflow. Stage 1 loads Titanic exactly once through Seaborn and commits `titanic.csv` as the offline fallback. Stage 2 reads that same CSV; it never calls `sns.load_dataset`.

## Run in order

From the repository root:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe .\analytics\01_eda.py
.\venv\Scripts\python.exe .\analytics\02_modeling.py
```

The first command needs internet only once for `sns.load_dataset('titanic')`. After `titanic.csv` exists, you can rerun the modeling step entirely offline.

## What is produced

- `titanic.csv`: committed raw offline fallback created immediately after the sole Seaborn load.
- `titanic_eda_cleaned.csv`: cleaned EDA dataset.
- `eda_report.md`: profiling output, missing-data decisions, survival breakdowns, correlation interpretation, outlier/skewness results, and chart interpretations.
- `modeling_report.md`: leakage controls, classifier metrics, imbalance comparison, grid-search/OOB result, regression metrics, and deployment recommendation.
- `plots/`: EDA, decision-tree, ROC, confusion-matrix, and residual plots.
- `best_classifier_pipeline.joblib`: full fitted preprocessing + tuned Random Forest pipeline, verified after reload on raw rows.

## Key design decisions

Rows with under-5% missing values in `embarked`/`embark_town` are dropped. `age` (5–30% missing) is median-imputed, while high-missing `deck` is represented with an explicit `Missing` category. The EDA report records the exact measured percentages.

Model preprocessing is deliberately independent from EDA and structurally leakage-safe: imputers, encoder, and scaler are contained in a scikit-learn `Pipeline`/`ColumnTransformer`, fitted only on the stratified training split. The imbalance experiment applies SMOTE only after transforming training data, never the test set.
