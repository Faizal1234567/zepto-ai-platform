# Titanic EDA report

## Load and profile

`sns.load_dataset('titanic')` is called exactly once in `01_eda.py`, then saved immediately as the committed offline fallback `titanic.csv`. Initial shape: **891 rows × 15 columns**. `df.info()` and `df.describe()` are printed during the run; the concise numeric profile is below.

| column | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| survived | 891.0 | 0.384 | 0.487 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| pclass | 891.0 | 2.309 | 0.836 | 1.0 | 2.0 | 3.0 | 3.0 | 3.0 |
| age | 714.0 | 29.699 | 14.526 | 0.42 | 20.125 | 28.0 | 38.0 | 80.0 |
| sibsp | 891.0 | 0.523 | 1.103 | 0.0 | 0.0 | 0.0 | 1.0 | 8.0 |
| parch | 891.0 | 0.382 | 0.806 | 0.0 | 0.0 | 0.0 | 0.0 | 6.0 |
| fare | 891.0 | 32.204 | 49.693 | 0.0 | 7.91 | 14.454 | 31.0 | 512.329 |

## Missing-value decisions

| column | missing_percent |
| --- | --- |
| deck | 77.22 |
| age | 19.87 |
| embarked | 0.22 |
| embark_town | 0.22 |

`embarked` and redundant `embark_town` have under 5% missing values, so their two incomplete rows are dropped under the stated threshold rule. `age` has 19.87% missingness (between 5% and 30%), so it is median-imputed. `deck` has 77.22% missingness, too high for reliable imputation; it is retained as a categorical feature with an explicit `Missing` level. The EDA-cleaned result contains **889 rows** and is saved as `titanic_eda_cleaned.csv`.

## Univariate findings

- Age IQR boundaries use Q1=22.00, Q3=35.00; **65** observations are outliers.
- Fare IQR boundaries use Q1=7.90, Q3=31.00; **114** observations are outliers.
- Fare mean=32.10, median=14.45, mode=8.05. The mean > median > mode ordering identifies a **right-skewed** fare distribution.

## Bivariate survival rates

### By sex
| sex | survived |
| --- | --- |
| female | 0.74 |
| male | 0.189 |

### By passenger class
| pclass | survived |
| --- | --- |
| 1 | 0.626 |
| 2 | 0.473 |
| 3 | 0.242 |

### By sex and passenger class
| sex | pclass | survived |
| --- | --- | --- |
| female | 1 | 0.967 |
| female | 2 | 0.921 |
| female | 3 | 0.5 |
| male | 1 | 0.369 |
| male | 2 | 0.157 |
| male | 3 | 0.135 |

Boolean-mask checks: female=0.740, male=0.189, female first class (`&`)=0.967, male third class (`&`)=0.135, and female **or** first class (`|`)=0.636.

## Correlation analysis

The matrix intentionally contains exactly `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`; derived boolean fields `adult_male` and `alone` are excluded. 

| feature | survived | pclass | age | sibsp | parch | fare |
| --- | --- | --- | --- | --- | --- | --- |
| survived | 1.0 | -0.336 | -0.07 | -0.034 | 0.083 | 0.255 |
| pclass | -0.336 | 1.0 | -0.337 | 0.082 | 0.017 | -0.548 |
| age | -0.07 | -0.337 | 1.0 | -0.233 | -0.171 | 0.094 |
| sibsp | -0.034 | 0.082 | -0.233 | 1.0 | 0.415 | 0.161 |
| parch | 0.083 | 0.017 | -0.171 | 0.415 | 1.0 | 0.218 |
| fare | 0.255 | -0.548 | 0.094 | 0.161 | 0.218 | 1.0 |

The largest absolute off-diagonal correlations are **pclass ↔ fare (-0.548)** and **sibsp ↔ parch (0.415)**. The first reflects how ticket cost and social class travel together; the second indicates the corresponding relationship in the numeric Titanic features, not a causal claim.

## Multivariate data story

1. `plots/survival_by_sex.png` shows a pronounced female survival advantage. This separation is much larger than the variation within either sex. Sex is therefore likely to be a highly informative classifier feature.
2. `plots/survival_by_pclass.png` shows survival decreasing from first to third class. Class likely proxies for cabin location, access, and socioeconomic advantage. It should be interpreted jointly with fare because the two are correlated.
3. `plots/survival_by_sex_pclass.png` shows the sex advantage across classes, while first-class women have the highest observed survival. Third-class men have the lowest rate. This interaction motivates retaining both features rather than relying on a single aggregate.
4. `plots/age_fare_survival.png` shows substantial fare spread and overlapping ages between outcomes. Survival does not separate cleanly by age or fare alone. Combining multiple features is therefore more appropriate than a one-variable decision rule.

## Exploratory standardization check

| feature | before_mean | before_std | z_mean | z_std |
| --- | --- | --- | --- | --- |
| age | 29.3152 | 12.9776 | 0.0 | 1.0 |
| fare | 32.0967 | 49.6695 | 0.0 | 1.0 |

The z-score columns have approximately zero mean and unit population standard deviation. This full-data transformation is EDA-only and is not used for modeling; the later model pipeline fits its scaler on training data only.
