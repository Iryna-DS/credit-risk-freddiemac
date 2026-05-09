# Mortgage Default Risk Model
### Freddie Mac Single-Family Loan-Level Dataset (2022–2023)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![sklearn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

---

## Business Problem

Freddie Mac purchases mortgage loans from lenders across the US.
Identifying loans at risk of default **at origination** allows portfolio
managers to apply enhanced monitoring, adjust pricing, or set reserves proactively.

**Goal:** Build a model that predicts probability of mortgage default
(90+ days delinquency) using only information available at loan origination.

---

## Dataset

| | |
|---|---|
| **Source** | Freddie Mac Single Family Loan-Level Dataset |
| **Period** | 2022–2023 (sample, 50K loans per year) |
| **Size** | 100,000 loans, 24 features |
| **Target** | `default` = loan reached 90+ days delinquency |
| **Default rate** | 2.43% (40:1 class imbalance) |
| **Download** | [freddiemac.com](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset) |

---

## Results

| Model | Test AUC | CV AUC (5-fold) |
|-------|----------|-----------------|
| Logistic Regression | **0.7950** | **0.8028 ± 0.006** |
| Gradient Boosting | 0.7922 | 0.8005 ± 0.009 |
| Random Forest | 0.7870 | 0.7926 ± 0.008 |
| LR Tuned (GridSearch) | 0.7947 | — |
| RF Tuned (RandomSearch) | 0.7870 | — |

**Selected model:** Logistic Regression — highest AUC, most stable, interpretable coefficients.  
AUC ~0.79–0.80 is consistent with mortgage default predictability at origination.

### Business Impact (test set, threshold = 0.60)
- Default capture rate: **62.6%**
- Estimated net savings: **~$18M** (vs no model baseline)
- Assumptions: 35% loss rate on defaulted UPB, $500/manual review cost

---

## Key Findings

### EDA
- **FICO score** is the strongest default predictor — 11% default rate for FICO 596–640 vs 0.4% for 801–832
- **Mobile Homes** default at 4.2% — nearly 2x portfolio average
- **Broker channel** shows highest default rate (2.63%) vs Correspondent (2.12%)
- Interest rates rose **+165bps** from 2022 to 2023, suppressing refinance activity
- Portfolio delinquency continues rising through 2025 — peak default not yet reached

### Modeling
- All models converge at AUC ~0.79 — reflects natural predictability ceiling at origination
- Low CV std (<0.01) confirms results are stable across different data splits
- `credit_score`, `orig_upb`, `dti`, `ocltv` are top predictive features
- class imbalance (40:1) handled via `class_weight='balanced'`

---

## Project Structure

```
credit-risk-freddiemac/
├── data/                              # Raw data files (not committed)
│   ├── sample_orig_2022.txt
│   ├── sample_orig_2023.txt
│   ├── sample_svcg_2022.txt
│   ├── sample_svcg_2023.txt
│   └── modeling_data.csv
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb   # Feature creation & selection
│   └── 03_model.ipynb                 # Modeling, evaluation & business impact
├── src/
│   └── utils.py                       # Data loading & preprocessing functions
├── requirements.txt
└── README.md
```


---

## Technical Stack

- **Python 3.9**
- **pandas**, **numpy** — data manipulation
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — modeling, evaluation, hyperparameter tuning
- **xgboost** — gradient boosting
