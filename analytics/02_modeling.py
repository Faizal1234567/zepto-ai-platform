"""Module 2, stage 2: train, compare, tune, and persist Titanic models.

Reads the offline CSV created by 01_eda.py; it never calls sns.load_dataset.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
RAW_CSV = HERE / "titanic.csv"
MODEL_PATH = HERE / "best_classifier_pipeline.joblib"
REPORT_PATH = HERE / "modeling_report.md"
RANDOM_STATE = 42


def table(frame: pd.DataFrame, digits: int = 3) -> str:
    result = frame.copy()
    for col in result.select_dtypes(include="number"):
        result[col] = result[col].round(digits)
    lines = ["| " + " | ".join(map(str, result.columns)) + " |", "| " + " | ".join(["---"] * len(result.columns)) + " |"]
    for row in result.fillna("").itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines)


def classification_pipeline(model: object) -> Pipeline:
    numeric = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical = ["sex", "embarked"]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", prep), ("model", model)])


def metrics(name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, float | str], np.ndarray, tuple[np.ndarray, np.ndarray]]:
    predicted = pipeline.predict(x_test)
    probability = pipeline.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probability)
    return ({"model": name, "accuracy": accuracy_score(y_test, predicted), "precision": precision_score(y_test, predicted, zero_division=0), "recall": recall_score(y_test, predicted, zero_division=0), "f1": f1_score(y_test, predicted, zero_division=0), "auc": auc(fpr, tpr)}, confusion_matrix(y_test, predicted), (fpr, tpr))


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError("Run 01_eda.py first to create analytics/titanic.csv.")
    PLOTS.mkdir(exist_ok=True)
    df = pd.read_csv(RAW_CSV)  # Offline continuation of Module 2's sole dataset load.
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    x, y = df[features], df["survived"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=RANDOM_STATE, stratify=y)
    class_balance = y.value_counts().rename_axis("survived").reset_index(name="count")
    class_balance["percent"] = class_balance["count"] / len(y) * 100

    models = {
        "Logistic Regression": classification_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "Decision Tree": classification_pipeline(DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        "Random Forest": classification_pipeline(RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
    }
    metric_rows, matrices, curves = [], {}, {}
    for name, pipe in models.items():
        pipe.fit(x_train, y_train)  # all imputation/encoding/scaling learns training data only
        row, matrix, curve = metrics(name, pipe, x_test, y_test)
        metric_rows.append(row); matrices[name] = matrix; curves[name] = curve
    comparison = pd.DataFrame(metric_rows)

    # Labeled decision-tree rendering from the training-only transformed features.
    tree_pipe = models["Decision Tree"]
    feature_names = tree_pipe.named_steps["preprocess"].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(tree_pipe.named_steps["model"], feature_names=feature_names, class_names=["not survived", "survived"], filled=True, rounded=True, ax=ax)
    fig.tight_layout(); fig.savefig(PLOTS / "decision_tree.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axes, models):
        sns.heatmap(matrices[name], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set(title=name, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout(); fig.savefig(PLOTS / "confusion_matrices.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (fpr, tpr) in curves.items():
        score = comparison.loc[comparison.model.eq(name), "auc"].iloc[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC={score:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray"); ax.legend(); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    fig.tight_layout(); fig.savefig(PLOTS / "roc_curves.png", dpi=150); plt.close(fig)

    # Imbalance comparison: SMOTE is fit only to train-fold transformed features.
    variants = {
        "Baseline": classification_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "class_weight=balanced": classification_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    }
    imbalance_rows = []
    for name, pipe in variants.items():
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        imbalance_rows.append({"strategy": name, "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0)})
    smote_preprocessor = classification_pipeline(LogisticRegression()).named_steps["preprocess"]
    train_transformed = smote_preprocessor.fit_transform(x_train)  # fit only on train
    test_transformed = smote_preprocessor.transform(x_test)
    x_resampled, y_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(train_transformed, y_train)
    smote_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(x_resampled, y_resampled)
    smote_pred = smote_model.predict(test_transformed)
    imbalance_rows.append({"strategy": "SMOTE (training fold only)", "precision": precision_score(y_test, smote_pred, zero_division=0), "recall": recall_score(y_test, smote_pred, zero_division=0), "f1": f1_score(y_test, smote_pred, zero_division=0)})
    imbalance = pd.DataFrame(imbalance_rows)

    # Grid search is entirely within training data and exposes the fitted model's OOB score.
    grid_pipe = classification_pipeline(RandomForestClassifier(oob_score=True, random_state=RANDOM_STATE, n_jobs=-1))
    grid = GridSearchCV(grid_pipe, {"model__n_estimators": [100, 200], "model__max_depth": [None, 8], "model__max_features": ["sqrt", "log2"]}, scoring="roc_auc", cv=3, n_jobs=-1)
    grid.fit(x_train, y_train)
    best_pipeline: Pipeline = grid.best_estimator_
    oob_score = best_pipeline.named_steps["model"].oob_score_
    # Persist the actual highest-F1 holdout model as one complete raw-input pipeline.
    deployment_pipeline = models[comparison.sort_values("f1", ascending=False).iloc[0]["model"]]
    joblib.dump(deployment_pipeline, MODEL_PATH)
    reloaded = joblib.load(MODEL_PATH)
    reload_matches = np.array_equal(deployment_pipeline.predict(x_test.iloc[:5]), reloaded.predict(x_test.iloc[:5]))

    # Regression predicts fare from other usable Titanic features; target is never a predictor.
    regression_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked", "class", "who", "adult_male", "alone"]
    rx, ry = df[regression_features], df["fare"]
    rx_train, rx_test, ry_train, ry_test = train_test_split(rx, ry, test_size=.2, random_state=RANDOM_STATE)
    r_num = ["pclass", "age", "sibsp", "parch"]
    r_cat = [c for c in regression_features if c not in r_num]
    r_preprocess = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), r_num),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), r_cat),
    ])
    regression = Pipeline([("preprocess", r_preprocess), ("model", LinearRegression())])
    regression.fit(rx_train, ry_train)
    fare_prediction = regression.predict(rx_test)
    r2 = r2_score(ry_test, fare_prediction); n, p = len(ry_test), regression.named_steps["preprocess"].transform(rx_test).shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    regression_metrics = {"MAE": mean_absolute_error(ry_test, fare_prediction), "RMSE": mean_squared_error(ry_test, fare_prediction) ** .5, "R2": r2, "Adjusted R2": adjusted_r2}
    residuals = ry_test - fare_prediction
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(fare_prediction, residuals, alpha=.65); ax.axhline(0, color="red", linestyle="--"); ax.set(xlabel="Predicted fare", ylabel="Residual", title="Fare regression residuals")
    fig.tight_layout(); fig.savefig(PLOTS / "regression_residuals.png", dpi=150); plt.close(fig)
    low_spread = residuals[fare_prediction <= np.median(fare_prediction)].std(); high_spread = residuals[fare_prediction > np.median(fare_prediction)].std()
    hetero = high_spread > low_spread * 1.25 or low_spread > high_spread * 1.25
    
    best_classification = comparison.sort_values("f1", ascending=False).iloc[0]
    final_table = pd.concat([
        comparison.assign(metric_group="Classification", MAE=np.nan, RMSE=np.nan, R2=np.nan, adjusted_R2=np.nan),
        pd.DataFrame([{"model": "Linear Regression (fare)", "metric_group": "Regression", "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "auc": np.nan, "MAE": regression_metrics["MAE"], "RMSE": regression_metrics["RMSE"], "R2": regression_metrics["R2"], "adjusted_R2": regression_metrics["Adjusted R2"]}]),
    ], ignore_index=True)
    report = f"""# Titanic modeling report

## Split and leakage controls

`02_modeling.py` reads the committed `titanic.csv` created by `01_eda.py`; it makes no second Seaborn/network load. The target class balance is below. A stratified 80/20 split is used before preprocessing because the survival outcome is not evenly distributed; stratification preserves that ratio in both splits and makes evaluation comparable.

{table(class_balance, 2)}

Each classification pipeline uses median imputation and `StandardScaler` for numeric columns, plus most-frequent imputation and one-hot encoding for `sex` and `embarked`. All transformations are fit in `Pipeline.fit(x_train, y_train)` only, then applied to the holdout with transform/predict. The regression pipeline follows the same training-only principle.

## Classifier evaluation

{table(comparison, 3)}

Confusion matrices are in `plots/confusion_matrices.png`; ROC curves/AUC are in `plots/roc_curves.png`; and the labeled decision tree is in `plots/decision_tree.png`.

## Imbalance comparison

{table(imbalance, 3)}

SMOTE is applied only after fitting the preprocessing transformer to `x_train`, and only resamples that transformed training fold. The best F1 in this run is **{imbalance.loc[imbalance.f1.idxmax(), 'strategy']}** ({imbalance.f1.max():.3f}); the precision/recall trade-off is why F1, rather than accuracy alone, is used for this comparison.

## Random Forest tuning

`GridSearchCV` searched `n_estimators`, `max_depth`, and `max_features` using three training folds. Best parameters: `{grid.best_params_}`; validation ROC-AUC: **{grid.best_score_:.3f}**; OOB score from `RandomForestClassifier(oob_score=True)`: **{oob_score:.3f}**.

## Fare regression

{table(pd.DataFrame([regression_metrics]), 3)}

`plots/regression_residuals.png` shows the residuals. The low- versus high-prediction residual standard deviations are {low_spread:.2f} and {high_spread:.2f}; this is {'evidence of heteroscedasticity (a non-random change in spread)' if hetero else 'not strong evidence of heteroscedasticity by this spread check'}.

## Separate model comparison and recommendation

{table(final_table, 3)}

The highest holdout F1 classifier is **{best_classification.model}** (accuracy={best_classification.accuracy:.3f}, precision={best_classification.precision:.3f}, recall={best_classification.recall:.3f}, F1={best_classification.f1:.3f}, AUC={best_classification.auc:.3f}). I would deploy it provisionally because F1 balances missed survivors and false-positive survivor predictions better than accuracy alone. The grid-search result and its OOB score provide an additional robustness check for the Random Forest family. Final selection should also consider the preferred operational trade-off between precision and recall.

## Saved complete pipeline

`best_classifier_pipeline.joblib` contains both the fitted `ColumnTransformer` preprocessing and the highest-F1 holdout classifier, {best_classification.model}—not a bare model. It was reloaded with `joblib.load` and made identical predictions on five raw, unprocessed test rows: **{reload_matches}**.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Trained three classifiers and fare regression. Saved {MODEL_PATH.name}, {REPORT_PATH.name}, and plots/.")
    print(comparison.to_string(index=False)); print("\nBest grid:", grid.best_params_, "OOB:", round(oob_score, 3)); print("Reload check:", reload_matches)


if __name__ == "__main__":
    main()
