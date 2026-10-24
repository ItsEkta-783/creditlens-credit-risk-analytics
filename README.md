# CreditLens | Credit-card default analytics
🔗 **Live Demo:** [Open CreditLens]([https://creditlens-credit-risk-analytics-zavsmll3txouzqx72laauz.streamlit.app/](https://creditlens-credit-risk-analytics-zavsmll3txouzqx72laauz.streamlit.app/))

CreditLens is a portfolio project for exploring historical credit-card repayment behavior, comparing baseline classification pipelines, and explaining their outputs. It is an educational analysis of **next-month credit-card default**; it does not predict loan approval or recommend credit decisions.

## Dataset

Uses the Default of Credit Card Clients dataset from the UCI Machine Learning Repository, containing 30,000 records and 23 explanatory features. The target indicates whether a client defaulted on payment the following month.

## Features

- Interactive dashboard for exploring credit-card repayment patterns.
- Logistic Regression and Random Forest model comparison.
- Held-out test evaluation using classification metrics, ROC, precision-recall, and calibration curves.
- Individual default-risk estimates and global/local SHAP explanations.
- Interactive filters and visualizations for customer demographics and repayment behavior.

## Methodology

The data is cleaned and split into stratified training (80%) and testing (20%) sets using random seed 42. Leakage-safe preprocessing pipelines train Logistic Regression and Random Forest models. Performance is evaluated on held-out data, with SHAP used to interpret Random Forest predictions.

## Findings and held-out evaluation

The workbook contains **30,000** records: **23,364 (77.88%)** are class 0 and **6,636 (22.12%)** are class 1. The held-out test partition contains 6,000 records, with a 22.12% default share.

Results below were reproduced with seed 42 and the pipelines in `src/modeling.py`. They are single-split estimates, not guarantees of future performance.

| Model | Accuracy | Precision | Recall | F1 | Specificity | ROC AUC | Average precision | Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7735 | 0.4893 | 0.5531 | 0.5193 | 0.8361 | 0.7588 | 0.5241 | 0.1865 |
| Random Forest | 0.7968 | 0.5412 | 0.5350 | 0.5381 | 0.8712 | 0.7747 | 0.5522 | 0.1580 |

At the illustrative 0.50 threshold, Logistic Regression has slightly higher default recall, while Random Forest has higher ROC AUC and average precision and lower Brier score on this split. Confusion matrices (rows = actual 0/1; columns = predicted 0/1): Logistic Regression `[[3907, 766], [593, 734]]`; Random Forest `[[4071, 602], [617, 710]]`.

## Limitations and responsible use

- The dataset reflects historical Taiwanese credit-card data and may not generalize to current populations.
- Evaluation uses a single random split; external validation and fairness assessments are not included.
- SHAP explanations show model behavior, not causation.
- This project is educational and is not intended for real-world lending decisions or automated credit approval.

## Setup and local run

**Requirements:** Python 3.11

Clone the repository and navigate to its root directory.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

Run the tests:

```bash
python -m pytest -q
```

The application loads the processed dataset when available or cleans the included source workbook. No external API keys are required.

## Project layout

```text
app.py             Streamlit application
src/clean_data.py  reusable workbook cleaning pipeline
src/modeling.py    split, preprocessing, models, held-out evaluation
data/              attributed source workbook and generated processed CSV
reports/           data dictionary, EDA report, and figures
tests/             cleaning, preprocessing, and evaluation tests
```
