# Titanic modeling report

## Split and leakage controls

`02_modeling.py` reads the committed `titanic.csv` created by `01_eda.py`; it makes no second Seaborn/network load. The target class balance is below. A stratified 80/20 split is used before preprocessing because the survival outcome is not evenly distributed; stratification preserves that ratio in both splits and makes evaluation comparable.

| survived | count | percent |
| --- | --- | --- |
| 0 | 549 | 61.62 |
| 1 | 342 | 38.38 |

Each classification pipeline uses median imputation and `StandardScaler` for numeric columns, plus most-frequent imputation and one-hot encoding for `sex` and `embarked`. All transformations are fit in `Pipeline.fit(x_train, y_train)` only, then applied to the holdout with transform/predict. The regression pipeline follows the same training-only principle.

## Classifier evaluation

| model | accuracy | precision | recall | f1 | auc |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| Decision Tree | 0.765 | 0.755 | 0.58 | 0.656 | 0.797 |
| Random Forest | 0.81 | 0.797 | 0.681 | 0.734 | 0.829 |

Confusion matrices are in `plots/confusion_matrices.png`; ROC curves/AUC are in `plots/roc_curves.png`; and the labeled decision tree is in `plots/decision_tree.png`.

## Imbalance comparison

| strategy | precision | recall | f1 |
| --- | --- | --- | --- |
| Baseline | 0.793 | 0.667 | 0.724 |
| class_weight=balanced | 0.73 | 0.783 | 0.755 |
| SMOTE (training fold only) | 0.74 | 0.783 | 0.761 |

SMOTE is applied only after fitting the preprocessing transformer to `x_train`, and only resamples that transformed training fold. The best F1 in this run is **SMOTE (training fold only)** (0.761); the precision/recall trade-off is why F1, rather than accuracy alone, is used for this comparison.

## Random Forest tuning

`GridSearchCV` searched `n_estimators`, `max_depth`, and `max_features` using three training folds. Best parameters: `{'model__max_depth': None, 'model__max_features': 'sqrt', 'model__n_estimators': 100}`; validation ROC-AUC: **0.866**; OOB score from `RandomForestClassifier(oob_score=True)`: **0.803**.

## Fare regression

| MAE | RMSE | R2 | Adjusted R2 |
| --- | --- | --- | --- |
| 18.724 | 30.85 | 0.385 | 0.311 |

`plots/regression_residuals.png` shows the residuals. The low- versus high-prediction residual standard deviations are 9.45 and 38.57; this is evidence of heteroscedasticity (a non-random change in spread).

## Separate model comparison and recommendation

| model | accuracy | precision | recall | f1 | auc | metric_group | MAE | RMSE | R2 | adjusted_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 | Classification |  |  |  |  |
| Decision Tree | 0.765 | 0.755 | 0.58 | 0.656 | 0.797 | Classification |  |  |  |  |
| Random Forest | 0.81 | 0.797 | 0.681 | 0.734 | 0.829 | Classification |  |  |  |  |
| Linear Regression (fare) |  |  |  |  |  | Regression | 18.724 | 30.85 | 0.385 | 0.311 |

The highest holdout F1 classifier is **Random Forest** (accuracy=0.810, precision=0.797, recall=0.681, F1=0.734, AUC=0.829). I would deploy it provisionally because F1 balances missed survivors and false-positive survivor predictions better than accuracy alone. The grid-search result and its OOB score provide an additional robustness check for the Random Forest family. Final selection should also consider the preferred operational trade-off between precision and recall.

## Saved complete pipeline

`best_classifier_pipeline.joblib` contains both the fitted `ColumnTransformer` preprocessing and the highest-F1 holdout classifier, Random Forest—not a bare model. It was reloaded with `joblib.load` and made identical predictions on five raw, unprocessed test rows: **True**.
