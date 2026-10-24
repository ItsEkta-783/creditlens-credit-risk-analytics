"""Generate a concise, reproducible EDA report and charts from cleaned data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.clean_data import ROOT, TARGET

DATA_FILE = ROOT / "data" / "processed" / "credit_default_clean.csv"
REPORT_FILE = ROOT / "reports" / "eda_summary.md"
FIGURES = ROOT / "reports" / "figures"


def generate(data: pd.DataFrame) -> str:
    target_counts = data[TARGET].value_counts().sort_index()
    total = len(data)
    default_rate = float(data[TARGET].mean())
    missing = int(data.isna().sum().sum())
    dupes = int(data.duplicated().sum())
    predictors = [c for c in data.columns if c not in {"id", TARGET}]
    description = data[predictors].describe().round(2)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = ["No default (0)", "Default (1)"]
    counts = [int(target_counts.get(0, 0)), int(target_counts.get(1, 0))]
    bars = ax.bar(labels, counts, color=["#3b82f6", "#f97316"])
    ax.set_title("Next-month payment default class distribution")
    ax.set_ylabel("Clients")
    ax.bar_label(bars, fmt="{:,}")
    fig.tight_layout()
    fig.savefig(FIGURES / "class_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    pay_columns = [f"pay_{n}" for n in range(1, 7)]
    means = data.groupby(TARGET)[pay_columns].mean().T
    means.columns = ["No default", "Default"]
    means.plot(kind="bar", ax=ax, color=["#3b82f6", "#f97316"])
    ax.set_title("Mean repayment status by outcome")
    ax.set_xlabel("Repayment status (monthly records)")
    ax.set_ylabel("Mean status code")
    ax.legend(title="Next-month outcome")
    fig.tight_layout()
    fig.savefig(FIGURES / "repayment_status_by_default.png", dpi=160)
    plt.close(fig)

    table = "| Statistic | " + " | ".join(description.columns) + " |\n"
    table += "|---|" + "|".join("---:" for _ in description.columns) + "|\n"
    for statistic, row in description.iterrows():
        table += f"| {statistic} | " + " | ".join(f"{value:,.2f}" for value in row) + " |\n"
    counts_md = "\n".join(
        f"| {int(cls)} | {int(count):,} | {count / total:.2%} |"
        for cls, count in target_counts.items()
    )
    return f"""# Credit card default: exploratory summary

## Dataset and preparation

- Source: `data/default of credit card clients.xls`, sheet `Data`.
- Header: second worksheet row (the first row contains X1–X23/Y aliases).
- Cleaned observations: **{total:,}**; retained columns: **{len(data.columns)}**.
- Identifier: `id` (excluded from predictor summaries); target: `{TARGET}`.
- Missing cells after cleaning: **{missing:,}**; duplicate rows after cleaning: **{dupes:,}**.
- Default rate: **{default_rate:.2%}**.

Cleaning preserves the source workbook. It normalizes field names, renames `PAY_0` to `pay_1`, drops duplicate records, and removes rows whose target is missing or not numeric. No imputation or feature encoding is performed.

## Target distribution

| Target value | Clients | Share |
|---:|---:|---:|
{counts_md}

![Target class distribution](figures/class_distribution.png)

## Numeric predictor summary

{table}

## Repayment status

Repayment status codes across the six historical monthly fields are shown as group means. The source uses negative codes as status categories; interpret these as codes rather than a continuous scale.

![Mean repayment status by target](figures/repayment_status_by_default.png)

## Scope

This report describes the supplied data only. It does not train or evaluate a predictive model and does not imply causal relationships.
"""


def main() -> None:
    data = pd.read_csv(DATA_FILE)
    report = generate(data)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_FILE.relative_to(ROOT)} and 2 figures from {len(data):,} rows.")


if __name__ == "__main__":
    main()
