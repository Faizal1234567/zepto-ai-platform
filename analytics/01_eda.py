"""Module 2, stage 1: load Titanic once, clean/profile it, and save EDA evidence."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
RAW_CSV = HERE / "titanic.csv"
CLEANED_CSV = HERE / "titanic_eda_cleaned.csv"
REPORT = HERE / "eda_report.md"
RANDOM_STATE = 42


def md_table(frame: pd.DataFrame, digits: int = 3) -> str:
    output = frame.copy()
    for col in output.select_dtypes(include="number"):
        output[col] = output[col].round(digits)
    columns = [str(c) for c in output.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in output.fillna("").itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in values) + " |")
    return "\n".join(lines)


def iqr_outliers(series: pd.Series) -> tuple[int, float, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()), q1, q3


def plot_eda(cleaned: pd.DataFrame, corr: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    for col in ["age", "fare"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(cleaned[col], kde=True, ax=axes[0], color="#2a6fbb")
        axes[0].set_title(f"{col.title()} distribution")
        sns.boxplot(x=cleaned[col], ax=axes[1], color="#79b7e7")
        axes[1].set_title(f"{col.title()} box plot")
        fig.tight_layout(); fig.savefig(PLOTS / f"{col}_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Specified numeric-feature correlation matrix")
    fig.tight_layout(); fig.savefig(PLOTS / "correlation_heatmap.png", dpi=150); plt.close(fig)

    chart_specs = [
        ("survival_by_sex.png", lambda ax: sns.barplot(data=cleaned, x="sex", y="survived", errorbar=None, ax=ax), "Survival rate by sex"),
        ("survival_by_pclass.png", lambda ax: sns.barplot(data=cleaned, x="pclass", y="survived", errorbar=None, ax=ax), "Survival rate by passenger class"),
        ("survival_by_sex_pclass.png", lambda ax: sns.barplot(data=cleaned, x="pclass", y="survived", hue="sex", errorbar=None, ax=ax), "Survival rate by sex and class"),
        ("age_fare_survival.png", lambda ax: sns.scatterplot(data=cleaned, x="age", y="fare", hue="survived", alpha=.7, ax=ax), "Age, fare, and survival"),
    ]
    for file_name, draw, title in chart_specs:
        fig, ax = plt.subplots(figsize=(8, 5)); draw(ax); ax.set_title(title)
        fig.tight_layout(); fig.savefig(PLOTS / file_name, dpi=150); plt.close(fig)


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    # The only network/cache dataset load anywhere in Module 2.
    df = sns.load_dataset("titanic")
    df.to_csv(RAW_CSV, index=False)  # Required offline fallback, saved immediately.
    initial_shape = df.shape
    missing = (df.isna().mean().mul(100)).loc[lambda s: s.gt(0)].sort_values(ascending=False)

    # Threshold decisions: <5% missing rows are dropped; 5-30% age is median-imputed;
    # deck is 77% missing and is retained with a deliberate Missing category.
    cleaned = df.dropna(subset=["embarked", "embark_town"]).copy()
    cleaned["age"] = cleaned["age"].fillna(cleaned["age"].median())
    cleaned["deck"] = cleaned["deck"].astype("string").fillna("Missing")
    CLEANED_CSV.write_text(cleaned.to_csv(index=False), encoding="utf-8")

    age_count, age_q1, age_q3 = iqr_outliers(cleaned["age"])
    fare_count, fare_q1, fare_q3 = iqr_outliers(cleaned["fare"])
    fare_mean, fare_median, fare_mode = cleaned["fare"].mean(), cleaned["fare"].median(), cleaned["fare"].mode().iloc[0]
    rates_sex = cleaned.groupby("sex", as_index=False)["survived"].mean()
    rates_class = cleaned.groupby("pclass", as_index=False)["survived"].mean()
    rates_both = cleaned.groupby(["sex", "pclass"], as_index=False)["survived"].mean()
    # Explicit boolean-mask calculations (also retained as values for the report).
    mask_rates = {
        "female": cleaned.loc[cleaned["sex"].eq("female"), "survived"].mean(),
        "male": cleaned.loc[cleaned["sex"].eq("male"), "survived"].mean(),
        "female_pclass_1": cleaned.loc[(cleaned["sex"].eq("female")) & (cleaned["pclass"].eq(1)), "survived"].mean(),
        "male_pclass_3": cleaned.loc[(cleaned["sex"].eq("male")) & (cleaned["pclass"].eq(3)), "survived"].mean(),
        "female_or_pclass_1": cleaned.loc[(cleaned["sex"].eq("female")) | (cleaned["pclass"].eq(1)), "survived"].mean(),
    }
    corr_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = cleaned[corr_columns].corr()
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    strongest = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(2)
    plot_eda(cleaned, corr)

    z = (cleaned[["age", "fare"]] - cleaned[["age", "fare"]].mean()) / cleaned[["age", "fare"]].std(ddof=0)
    z_summary = pd.DataFrame({"before_mean": cleaned[["age", "fare"]].mean(), "before_std": cleaned[["age", "fare"]].std(ddof=0), "z_mean": z.mean(), "z_std": z.std(ddof=0)})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ["age", "fare"]):
        sns.kdeplot(cleaned[col], ax=ax, label="before")
        sns.kdeplot(z[col], ax=ax, label="z-score")
        ax.set_title(f"{col.title()}: before and z-score transformed"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "zscore_before_after.png", dpi=150); plt.close(fig)

    report = f"""# Titanic EDA report

## Load and profile

`sns.load_dataset('titanic')` is called exactly once in `01_eda.py`, then saved immediately as the committed offline fallback `titanic.csv`. Initial shape: **{initial_shape[0]} rows × {initial_shape[1]} columns**. `df.info()` and `df.describe()` are printed during the run; the concise numeric profile is below.

{md_table(df.describe().T.reset_index().rename(columns={'index': 'column'}))}

## Missing-value decisions

{md_table(missing.rename('missing_percent').reset_index().rename(columns={'index': 'column'}), 2)}

`embarked` and redundant `embark_town` have under 5% missing values, so their two incomplete rows are dropped under the stated threshold rule. `age` has {missing['age']:.2f}% missingness (between 5% and 30%), so it is median-imputed. `deck` has {missing['deck']:.2f}% missingness, too high for reliable imputation; it is retained as a categorical feature with an explicit `Missing` level. The EDA-cleaned result contains **{len(cleaned)} rows** and is saved as `titanic_eda_cleaned.csv`.

## Univariate findings

- Age IQR boundaries use Q1={age_q1:.2f}, Q3={age_q3:.2f}; **{age_count}** observations are outliers.
- Fare IQR boundaries use Q1={fare_q1:.2f}, Q3={fare_q3:.2f}; **{fare_count}** observations are outliers.
- Fare mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}. The mean > median > mode ordering identifies a **right-skewed** fare distribution.

## Bivariate survival rates

### By sex
{md_table(rates_sex, 3)}

### By passenger class
{md_table(rates_class, 3)}

### By sex and passenger class
{md_table(rates_both, 3)}

Boolean-mask checks: female={mask_rates['female']:.3f}, male={mask_rates['male']:.3f}, female first class (`&`)={mask_rates['female_pclass_1']:.3f}, male third class (`&`)={mask_rates['male_pclass_3']:.3f}, and female **or** first class (`|`)={mask_rates['female_or_pclass_1']:.3f}.

## Correlation analysis

The matrix intentionally contains exactly `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`; derived boolean fields `adult_male` and `alone` are excluded. 

{md_table(corr.reset_index().rename(columns={'index': 'feature'}), 3)}

The largest absolute off-diagonal correlations are **{strongest.index[0][0]} ↔ {strongest.index[0][1]} ({strongest.iloc[0]:.3f})** and **{strongest.index[1][0]} ↔ {strongest.index[1][1]} ({strongest.iloc[1]:.3f})**. The first reflects how ticket cost and social class travel together; the second indicates the corresponding relationship in the numeric Titanic features, not a causal claim.

## Multivariate data story

1. `plots/survival_by_sex.png` shows a pronounced female survival advantage. This separation is much larger than the variation within either sex. Sex is therefore likely to be a highly informative classifier feature.
2. `plots/survival_by_pclass.png` shows survival decreasing from first to third class. Class likely proxies for cabin location, access, and socioeconomic advantage. It should be interpreted jointly with fare because the two are correlated.
3. `plots/survival_by_sex_pclass.png` shows the sex advantage across classes, while first-class women have the highest observed survival. Third-class men have the lowest rate. This interaction motivates retaining both features rather than relying on a single aggregate.
4. `plots/age_fare_survival.png` shows substantial fare spread and overlapping ages between outcomes. Survival does not separate cleanly by age or fare alone. Combining multiple features is therefore more appropriate than a one-variable decision rule.

## Exploratory standardization check

{md_table(z_summary.reset_index().rename(columns={'index': 'feature'}), 4)}

The z-score columns have approximately zero mean and unit population standard deviation. This full-data transformation is EDA-only and is not used for modeling; the later model pipeline fits its scaler on training data only.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"Loaded {initial_shape}; wrote {RAW_CSV.name}, {CLEANED_CSV.name}, {REPORT.name}, and plots/.")
    print(df.info())
    print(df.describe())


if __name__ == "__main__":
    main()
