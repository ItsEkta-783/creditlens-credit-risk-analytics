# Credit card default: exploratory summary

## Dataset and preparation

- Source: `data/default of credit card clients.xls`, sheet `Data`.
- Header: second worksheet row (the first row contains X1–X23/Y aliases).
- Cleaned observations: **30,000**; retained columns: **25**.
- Identifier: `id` (excluded from predictor summaries); target: `default_payment_next_month`.
- Missing cells after cleaning: **0**; duplicate rows after cleaning: **0**.
- Default rate: **22.12%**.

Cleaning preserves the source workbook. It normalizes field names, renames `PAY_0` to `pay_1`, drops duplicate records, and removes rows whose target is missing or not numeric. No imputation or feature encoding is performed.

## Target distribution

| Target value | Clients | Share |
|---:|---:|---:|
| 0 | 23,364 | 77.88% |
| 1 | 6,636 | 22.12% |

![Target class distribution](figures/class_distribution.png)

## Numeric predictor summary

| Statistic | limit_bal | sex | education | marriage | age | pay_1 | pay_2 | pay_3 | pay_4 | pay_5 | pay_6 | bill_amt1 | bill_amt2 | bill_amt3 | bill_amt4 | bill_amt5 | bill_amt6 | pay_amt1 | pay_amt2 | pay_amt3 | pay_amt4 | pay_amt5 | pay_amt6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| count | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 | 30,000.00 |
| mean | 167,484.32 | 1.60 | 1.85 | 1.55 | 35.49 | -0.02 | -0.13 | -0.17 | -0.22 | -0.27 | -0.29 | 51,223.33 | 49,179.08 | 47,013.15 | 43,262.95 | 40,311.40 | 38,871.76 | 5,663.58 | 5,921.16 | 5,225.68 | 4,826.08 | 4,799.39 | 5,215.50 |
| std | 129,747.66 | 0.49 | 0.79 | 0.52 | 9.22 | 1.12 | 1.20 | 1.20 | 1.17 | 1.13 | 1.15 | 73,635.86 | 71,173.77 | 69,349.39 | 64,332.86 | 60,797.16 | 59,554.11 | 16,563.28 | 23,040.87 | 17,606.96 | 15,666.16 | 15,278.31 | 17,777.47 |
| min | 10,000.00 | 1.00 | 0.00 | 0.00 | 21.00 | -2.00 | -2.00 | -2.00 | -2.00 | -2.00 | -2.00 | -165,580.00 | -69,777.00 | -157,264.00 | -170,000.00 | -81,334.00 | -339,603.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 25% | 50,000.00 | 1.00 | 1.00 | 1.00 | 28.00 | -1.00 | -1.00 | -1.00 | -1.00 | -1.00 | -1.00 | 3,558.75 | 2,984.75 | 2,666.25 | 2,326.75 | 1,763.00 | 1,256.00 | 1,000.00 | 833.00 | 390.00 | 296.00 | 252.50 | 117.75 |
| 50% | 140,000.00 | 2.00 | 2.00 | 2.00 | 34.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 22,381.50 | 21,200.00 | 20,088.50 | 19,052.00 | 18,104.50 | 17,071.00 | 2,100.00 | 2,009.00 | 1,800.00 | 1,500.00 | 1,500.00 | 1,500.00 |
| 75% | 240,000.00 | 2.00 | 2.00 | 2.00 | 41.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 67,091.00 | 64,006.25 | 60,164.75 | 54,506.00 | 50,190.50 | 49,198.25 | 5,006.00 | 5,000.00 | 4,505.00 | 4,013.25 | 4,031.50 | 4,000.00 |
| max | 1,000,000.00 | 2.00 | 6.00 | 3.00 | 79.00 | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 | 964,511.00 | 983,931.00 | 1,664,089.00 | 891,586.00 | 927,171.00 | 961,664.00 | 873,552.00 | 1,684,259.00 | 896,040.00 | 621,000.00 | 426,529.00 | 528,666.00 |


## Repayment status

Repayment status codes across the six historical monthly fields are shown as group means. The source uses negative codes as status categories; interpret these as codes rather than a continuous scale.

![Mean repayment status by target](figures/repayment_status_by_default.png)

## Scope

This report describes the supplied data only. It does not train or evaluate a predictive model and does not imply causal relationships.
