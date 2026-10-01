"""German credit risk: Logistic Regression vs XGBoost, with SHAP and a cost-based cut-off.

Target: default = 1 when the customer is a bad credit risk.
The original dataset documentation puts a 5:1 cost on approving a bad
customer versus rejecting a good one, so the approval cut-off is chosen
to minimise that cost instead of using the default 0.5.
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
COST_FN, COST_FP = 5, 1  # approving a defaulter costs 5, rejecting a good customer costs 1


def load():
    df = pd.read_csv("data/GermanCredit.csv")
    y = (df["credit_risk"] == 0).astype(int)  # in this file 0 = bad, 1 = good
    return df.drop(columns="credit_risk"), y


def total_cost(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return int(fn * COST_FN + fp * COST_FP)


def best_threshold(y, proba):
    grid = np.arange(0.05, 0.95, 0.01)
    costs = [total_cost(y, (proba >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmin(costs))])


def main():
    X, y = load()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = X.select_dtypes("number").columns.tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    prep_lr = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat),
                                 ("n", StandardScaler(), num)])
    prep_xgb = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat)],
                                 remainder="passthrough")
    models = {
        "logistic_regression": Pipeline([("prep", prep_lr),
                                         ("clf", LogisticRegression(max_iter=2000, C=0.5))]),
        "xgboost": Pipeline([("prep", prep_xgb),
                             ("clf", XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                                                   subsample=0.8, colsample_bytree=0.8,
                                                   eval_metric="logloss", random_state=SEED))]),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results = {}
    for name, pipe in models.items():
        oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        thr = best_threshold(y_tr, oof)  # chosen on training data only
        results[name] = {
            "cv_roc_auc": round(float(roc_auc_score(y_tr, oof)), 3),
            "test_roc_auc": round(float(roc_auc_score(y_te, proba)), 3),
            "chosen_threshold": round(thr, 2),
            "test_cost_at_0.5": total_cost(y_te, (proba >= 0.5).astype(int)),
            "test_cost_at_chosen_threshold": total_cost(y_te, (proba >= thr).astype(int)),
        }
        results[name]["cost_reduction_pct"] = round(
            (results[name]["test_cost_at_0.5"] - results[name]["test_cost_at_chosen_threshold"])
            / results[name]["test_cost_at_0.5"] * 100, 1)

    # SHAP on the XGBoost model
    xgb = models["xgboost"]
    Xt = xgb.named_steps["prep"].transform(X_te)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    names = [n.split("__")[-1] for n in xgb.named_steps["prep"].get_feature_names_out()]
    sv = shap.TreeExplainer(xgb.named_steps["clf"]).shap_values(Xt)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
    results["top_shap_features"] = imp.head(8).round(3).to_dict()
    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=120)

    best = max(["logistic_regression", "xgboost"], key=lambda k: results[k]["cv_roc_auc"])
    results["best_model_by_cv_auc"] = best
    joblib.dump({"model": models[best], "threshold": results[best]["chosen_threshold"],
                 "columns": X.columns.tolist()}, "model.joblib")
    json.dump(results, open("results.json", "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
