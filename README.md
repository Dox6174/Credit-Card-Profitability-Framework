# Credit-Card-Profitability-Framework

An economically-grounded, closed-form profitability scoring framework built for a 500,000-row, 23-column Amex credit card dataset. Round 1 score: **0.93** on both the public and private leaderboards.

Rather than fitting a black-box model to an unobserved profitability target, this project treats profitability as a deterministic accounting identity — revenue minus cost — so every coefficient in the final score is tied to a specific, interpretable economic mechanism rather than an opaque fitted weight.

## Contents

| File | Description |
|---|---|
| [`Credit_Card_Profitability_Framework_Report.pdf`](./Credit_Card_Profitability_Framework_Report.pdf) | Full research report: dataset profiling, EDA findings, correlation analysis, term-sheet-derived economic reasoning, and term-by-term coefficient justification for the final framework. |
| [`Exploratory_DA.ipynb`](./Exploratory_DA.ipynb) | Exploratory data analysis notebook — missingness structure, spend decomposition checks, Spearman correlation matrix, Pareto concentration curves, card-type segmentation, and exploratory modeling (PCA/GMM, proxy scoring, XGBoost ranker) that informed the final framework. |

## Final Framework

The score for each account is the sum of five terms — three revenue, two cost/loss:

```python
import pandas as pd
import numpy as np

def build_score(df: pd.DataFrame) -> pd.Series:
    """
    Computes the profitability score based strictly on the provided P&L equation.
    """
    f1 = df['f1'].fillna(0)
    f3 = df['f3'].fillna(0)
    f6 = df['f6'].fillna(0)
    f7 = df['f7'].fillna(0)
    f8 = df['f8'].fillna(0)
    f9 = df['f9'].fillna(0)
    f10 = df['f10'].fillna(0)
    f11 = df['f11'].fillna(0)
    f13 = df['f13'].fillna(0)
    f14 = df['f14'].fillna(0)
    f15 = df['f15'].fillna(0)
    f16 = df['f16'].fillna(0)
    f17 = df['f17'].fillna(0)
    f19 = df['f19'].fillna(0)

    f11 = f11 + 0.2 * f3

    wallet_spend = f6 + f7 + f8 + f9 + f10
    wallet_spend = np.clip(wallet_spend, 0, None)

    term_transaction = 5075.4 * ((wallet_spend / 203700.0) ** 0.20)
    term_interest = 0.21 * f1
    term_loss_drawn = 0.55 * f11 * f1
    term_loss_undrawn = 0.055 * f11 * f17
    term_benefits = (50.0 * f13) + f14 + (15.0 * f15) + f16

    score = term_transaction \
          + term_interest \
          - term_loss_drawn \
          - term_loss_undrawn \
          - term_benefits
    return score
```

### Why this structure

- **Transaction value** — a concave power-law function of aggregated wallet spend, reflecting diminishing marginal revenue at high spend levels (supported by Pareto concentration analysis in the notebook).
- **Interest margin** — a flat rate applied to drawn balance, calibrated to the midpoint of an assumed APR range.
- **Expected loss (drawn / undrawn)** — a shared loss-given-default rate applied to drawn balance at full severity and to undrawn line at a 10% credit-conversion-factor discount, with the risk indicator boosted using a collections-call signal shown (via Spearman correlation) to carry incremental risk information.
- **Benefit cost-to-serve** — a direct, usage-weighted dollar sum over the four premium benefit fields.

Full derivations, the correlation evidence behind each design choice, and the term-sheet economics the framework was distilled from are documented in the report.

## Result

| Split | Score |
|---|---|
| Public leaderboard | 0.93 |
| Private leaderboard | 0.93 |

The agreement between public and private scores reflects the framework's closed-form structure — since the score is a deterministic function of input features rather than a fitted model, it carries no risk of overfitting to the public split.

## Reproducing

1. Open `tech-support.ipynb` to reproduce the exploratory data analysis.
2. Apply `build_score()` above to the raw dataset to reproduce the Round 1 submission scores.
