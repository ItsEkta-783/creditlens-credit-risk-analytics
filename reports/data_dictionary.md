# Data dictionary

The workbook sheet is `Data`. Worksheet row 2 contains field names; row 1 contains aliases (`X1`–`X23`, `Y`). The descriptions below reflect the dataset's field labels and domains. Monetary fields are in the source dataset's stated NT dollar units.

| Source field | Clean field | Meaning / values |
|---|---|---|
| ID | id | Client identifier; not a predictor. |
| LIMIT_BAL | limit_bal | Credit limit amount. |
| SEX | sex | Gender category: 1=male, 2=female. |
| EDUCATION | education | Education category: 1=graduate school, 2=university, 3=high school, 4=others; 0, 5, 6 also occur as source codes. |
| MARRIAGE | marriage | Marital status: 1=married, 2=single, 3=others; 0 also occurs. |
| AGE | age | Age in years. |
| PAY_0 | pay_1 | Repayment status in September 2005; codes include -2, -1, 0 and positive delay-month counts. |
| PAY_2 | pay_2 | Repayment status in August 2005. |
| PAY_3 | pay_3 | Repayment status in July 2005. |
| PAY_4 | pay_4 | Repayment status in June 2005. |
| PAY_5 | pay_5 | Repayment status in May 2005. |
| PAY_6 | pay_6 | Repayment status in April 2005. |
| BILL_AMT1 | bill_amt1 | September 2005 bill statement amount. |
| BILL_AMT2 | bill_amt2 | August 2005 bill statement amount. |
| BILL_AMT3 | bill_amt3 | July 2005 bill statement amount. |
| BILL_AMT4 | bill_amt4 | June 2005 bill statement amount. |
| BILL_AMT5 | bill_amt5 | May 2005 bill statement amount. |
| BILL_AMT6 | bill_amt6 | April 2005 bill statement amount. |
| PAY_AMT1 | pay_amt1 | Payment amount in September 2005. |
| PAY_AMT2 | pay_amt2 | Payment amount in August 2005. |
| PAY_AMT3 | pay_amt3 | Payment amount in July 2005. |
| PAY_AMT4 | pay_amt4 | Payment amount in June 2005. |
| PAY_AMT5 | pay_amt5 | Payment amount in May 2005. |
| PAY_AMT6 | pay_amt6 | Payment amount in April 2005. |
| default payment next month | default_payment_next_month | Target: next-month default indicator, 1=yes and 0=no. |

There are 23 predictors if `id` is excluded: 5 demographic/credit attributes, 6 repayment-status fields, 6 statement amounts, and 6 payment amounts. No model is included in this stage.
