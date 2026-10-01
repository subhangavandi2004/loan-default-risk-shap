# Loan Default Risk Model with SHAP

Predicts whether a loan applicant is a bad credit risk, explains the drivers with SHAP, and serves the model with FastAPI.

**Data:** German Credit, 1,000 applicants, 20 features, 30% bad risks
(https://github.com/selva86/datasets, `GermanCredit.csv`).

**Run:** `pip install -r requirements.txt`, `python train.py`, `pytest test_api.py`, `uvicorn app:app`.

## Method
- 80/20 stratified split, 5-fold CV on the training part only.
- Logistic Regression baseline vs XGBoost. Final model chosen by CV AUC, not by test score.
- Approval cut-off chosen on training predictions to minimise cost, using the dataset's 5:1 cost for approving a defaulter vs rejecting a good customer.

## Results

| Model | CV ROC-AUC | Test ROC-AUC | Test cost at 0.5 | Test cost at chosen cut-off | Cost reduction |
|---|---|---|---|---|---|
| Logistic Regression | 0.774 | 0.805 | 154 | 113 (cut-off 0.14) | 26.6% |
| **XGBoost** | **0.790** | 0.799 | 161 | 112 (cut-off 0.10) | **30.4%** |

Top SHAP drivers: no checking account, loan duration, loan amount, critical credit history, low savings.

## Limitations
The test set has only 200 rows, so the two models are effectively tied on test AUC (0.805 vs 0.799).
Cost numbers are in the dataset's own cost units, not money.

Files: `train.py`, `app.py`, `test_api.py`, `results.json`, `shap_summary.png`.
