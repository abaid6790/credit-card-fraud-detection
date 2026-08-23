# Dataset

## Source
This project uses the **Credit Card Fraud Detection** dataset published by the
Machine Learning Group at ULB (Université Libre de Bruxelles) on Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Structure
- **Rows:** 284,807 transactions made by European cardholders over two days in September 2013
- **Fraud cases:** 492 (0.172% of all transactions) — highly imbalanced
- **Columns:**
  - `Time` — seconds elapsed between this transaction and the first transaction in the dataset
  - `V1` … `V28` — principal components obtained via PCA; original features are not
    disclosed for confidentiality reasons
  - `Amount` — transaction amount
  - `Class` — target column: `0` = legitimate, `1` = fraud

## Target column
`Class` is the label. It must never be used as an input feature during training or inference.

## How to download
1. Create a free Kaggle account if you don't have one.
2. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
3. Click **Download** (or use the Kaggle CLI: `kaggle datasets download -d mlg-ulb/creditcardfraud`).
4. Unzip the archive — you'll get `creditcard.csv`.

## How to place it in this project
Put the file here:

```
data/creditcard.csv
```

It is intentionally excluded from Git via `.gitignore` — never commit the raw dataset.

## Limitations
- Features `V1`–`V28` are anonymized PCA components, so individual feature meaning
  cannot be interpreted directly (only relative importance/contribution).
- The data covers only two days from one European payment processor in 2013 — it
  does not reflect current fraud patterns, other geographies, or other card networks.
- The dataset has no merchant, device, geolocation, or customer-history data, which
  a real production fraud system would use.
- This dataset is intended for research/education. A model trained on it should be
  treated as a **fraud-risk estimator**, not a certified fraud-detection system.
