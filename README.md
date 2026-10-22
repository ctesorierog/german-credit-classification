# German Credit risk classification

Predicts whether a bank client is a good or bad credit risk using the **German Credit dataset** (credit-g, OpenML id 31). The project compares Logistic Regression, a linear SVM and k-Nearest Neighbours inside leakage-safe scikit-learn pipelines, and includes a written report.

## Data

1,000 loan applicants, each described by 20 features (7 numeric, 13 categorical). The target is good (700) or bad (300) credit risk, with no missing values. The notebook loads the data directly from OpenML with `fetch_openml(data_id=31)`, so no data file is needed.

## Method

- Exploratory analysis: class balance, histograms and boxplots of the numeric features by class, and every categorical feature against the target.
- Stratified 80/20 train/test split. The test set is used only once, at the end.
- `ColumnTransformer` pipeline: scaling for numeric features and one-hot encoding for categorical ones, fitted only on training data.
- Hyper-parameters tuned with `GridSearchCV` (cross-validation on the training set).

## Results (test set, bad risk = positive class)

| Model | Best parameter | CV accuracy | Test accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | C = 0.1 | 0.758 | 0.775 | 0.692 | 0.450 | 0.545 |
| **Linear SVM** | C = 0.1 | 0.761 | **0.785** | 0.698 | 0.500 | **0.583** |
| kNN | k = 7 | 0.740 | 0.735 | 0.630 | 0.283 | 0.391 |

The linear SVM performed best overall. All three models find recall on bad risks hard, which matters most for a bank. The report discusses this trade-off and how class weighting or a lower decision threshold could address it.

## Files

- `german_credit_classification.ipynb`: full analysis with all outputs.
- `docs/report.pdf`: written report covering background, method, results and discussion.

## Stack

Python, pandas, scikit-learn, seaborn, matplotlib, Google Colab.

## Context

Project 1, *Machine Learning*, AI Engineering, Tecnun – Universidad de Navarra, March 2026.
