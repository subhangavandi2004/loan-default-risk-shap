"""FastAPI service for the credit risk model. Run: uvicorn app:app --reload"""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

bundle = joblib.load("model.joblib")
app = FastAPI(title="Credit Risk API")


class Applicant(BaseModel):
    features: dict  # same columns as data/GermanCredit.csv, without credit_risk


@app.get("/health")
def health():
    return {"status": "ok", "threshold": bundle["threshold"]}


@app.post("/predict")
def predict(applicant: Applicant):
    missing = [c for c in bundle["columns"] if c not in applicant.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing fields: {missing}")
    row = pd.DataFrame([applicant.features])[bundle["columns"]]
    p = float(bundle["model"].predict_proba(row)[0, 1])
    return {"default_probability": round(p, 4),
            "decision": "reject" if p >= bundle["threshold"] else "approve"}
