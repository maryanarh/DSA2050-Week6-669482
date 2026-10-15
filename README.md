# DSA 2050 – Week 6: Regression and Baseline Comparison

**Student Name:** Mark Mayana
**Student ID:** 669482

## Objective
Build, evaluate, and compare regression models against baselines to estimate
`Sales_KES` from business characteristics, following a leakage-safe methodology.
The lab emphasises sound methodology, leakage-safe preprocessing, training-only
cross-validation, residual analysis, subgroup error checks, and interpretation.

## Dataset
- `DSA2050_Week6_Sales_Data.csv`
- 320 rows, 7 columns
- Target: `Sales_KES`
- Features: `AdvertisingSpend_KES`, `StoreVisits`, `DiscountPercent`,
  `Region`, `CustomerSegment`, `ProductCategory`
- No missing values, no duplicate rows

## Methodology
- 80/20 train/test split (random_state = 42)
- Leakage-safe preprocessing using `Pipeline` + `ColumnTransformer`:
  - Numeric: median imputation + standard scaling
  - Categorical: most-frequent imputation + one-hot encoding
- Model development on training data only; test set used once as a final check
- 5-fold cross-validation performed on the training set only

## Models Compared
- Mean Baseline (`DummyRegressor`)
- Median Baseline (`DummyRegressor`)
- Linear Regression
- Ridge Regression (alpha = 1.0)
- Shallow Decision Tree (max_depth = 4, min_samples_leaf = 8)

## Best Model
**Ridge Regression** — lowest MAE, lowest RMSE, and highest R² on both the
held-out test set and 5-fold cross-validation. Linear Regression was virtually
tied, and the shallow Decision Tree was clearly weaker.

## Selected Model MAE
Approximately **13,714 KES** on the held-out test set.

## Test-Set Comparison (MAE / RMSE / R²)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Ridge Regression | 13,714 | 17,459 | 0.8705 |
| Linear Regression | 13,758 | 17,470 | 0.8703 |
| Decision Tree | 23,776 | 28,768 | 0.6484 |
| Median Baseline | 39,147 | 48,649 | −0.006 |
| Mean Baseline | 39,197 | 48,716 | −0.008 |

## 5-Fold Cross-Validation (Training Set Only)
| Model | CV MAE | CV RMSE | CV R² |
|---|---|---|---|
| Ridge Regression | 15,971 | 20,104 | 0.8479 |
| Linear Regression | 15,977 | 20,127 | 0.8475 |
| Decision Tree | 24,498 | 30,609 | 0.6506 |

## Two Key Findings
1. All fitted models clearly beat the mean and median baselines, confirming that
   advertising spend, store visits, discount, region, segment, and product
   category carry real predictive signal. Ridge Regression improved on the
   baseline MAE by roughly 25,500 KES.
2. The unconstrained decision tree overfits dramatically (training MAE = 0 vs
   test MAE ≈ 26,798), while the shallow tree (max_depth = 4) generalises
   reasonably (training MAE ≈ 18,030 vs test MAE ≈ 23,776).

## Segment Error Analysis
Segment-level MAE on the test set (highest to lowest):
- Retail ≈ 15,561 KES (27 observations)
- Corporate ≈ 14,300 KES (19 observations)
- SME ≈ 10,481 KES (18 observations)

## Three Limitations
1. Small dataset (320 rows) limits model stability and generalisation; a larger
   sample could meaningfully change the coefficient estimates.
2. Limited predictors: the model ignores seasonality, competitor pricing,
   promotions, macroeconomic conditions, and store-level attributes that
   strongly influence sales.
3. Prediction is not causation: the model shows association, not cause-and-effect.
   Increasing advertising spend may not directly raise sales if both are driven
   by a third factor.

## How to Run
1. Open `DSA2050_Week6_Guided_Regression_Lab.ipynb` in Jupyter.
2. Ensure `DSA2050_Week6_Sales_Data.csv` is in the same folder.
3. Kernel → Restart & Run All to reproduce every table and figure.

## Repository Structure