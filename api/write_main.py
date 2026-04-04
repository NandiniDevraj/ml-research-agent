code = '''import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "champion_model.pkl")
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Champion model loaded")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="LightGBM model built by autonomous ML agent pipeline.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    age: float
    num_medications: float
    num_procedures: float
    num_diagnoses: float
    time_in_hospital: float
    num_lab_tests: float
    gender: float
    admission_type: float
    discharge_to: float
    primary_diagnosis: float
    A1C_result: float
    medication_intensity: float
    procedure_burden: float
    is_elderly: float
    log_num_medications: float

class PredictionResponse(BaseModel):
    readmission_risk: str
    readmission_probability: float
    confidence: str
    recommendation: str
    model_version: str

@app.get("/")
def root():
    return {"service": "Hospital Readmission API", "status": "live", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features = [
            "age", "num_medications", "num_procedures", "num_diagnoses",
            "time_in_hospital", "num_lab_tests", "gender", "admission_type",
            "discharge_to", "primary_diagnosis", "A1C_result",
            "medication_intensity", "procedure_burden", "is_elderly",
            "log_num_medications"
        ]
        input_data = np.array([[getattr(patient, f) for f in features]])
        probability = float(model.predict_proba(input_data)[0][1])
        prediction = "HIGH RISK" if probability >= 0.5 else "LOW RISK"
        if probability >= 0.7:
            confidence = "High"
            recommendation = "Immediate follow-up within 7 days"
        elif probability >= 0.5:
            confidence = "Medium"
            recommendation = "Schedule follow-up within 14 days"
        else:
            confidence = "High"
            recommendation = "Standard discharge routine follow-up"
        return PredictionResponse(
            readmission_risk=prediction,
            readmission_probability=round(probability, 4),
            confidence=confidence,
            recommendation=recommendation,
            model_version="LightGBM-v1-Champion"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
def model_info():
    return {
        "model_type": "LightGBM",
        "feature_count": 15,
        "performance": {"auc": 0.9177, "accuracy": 0.92},
        "pipeline": "6-agent autonomous ML system"
    }
'''

with open("main.py", "w", encoding="utf-8") as f:
    f.write(code)

print("main.py written successfully")