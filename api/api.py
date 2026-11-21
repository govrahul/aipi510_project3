import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

API_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = API_DIR / "water_potability_pipeline.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

RAW_INPUT_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

class WaterSample(BaseModel):
    ph: Optional[float] = Field(default=None, description="pH value of water")
    Hardness: Optional[float] = Field(default=None, description="Hardness in mg/L")
    Solids: Optional[float] = Field(default=None, description="Total dissolved solids in ppm")
    Chloramines: Optional[float] = Field(default=None, description="Chloramines in ppm")
    Sulfate: Optional[float] = Field(default=None, description="Sulfate in mg/L")
    Conductivity: Optional[float] = Field(default=None, description="Electrical conductivity in uS/cm")
    Organic_carbon: Optional[float] = Field(default=None, description="Total organic carbon in ppm")
    Trihalomethanes: Optional[float] = Field(default=None, description="Trihalomethanes in ug/L")
    Turbidity: Optional[float] = Field(default=None, description="Turbidity in NTU")

class PredictionResponse(BaseModel):
    potable: int
    label: str
    probability_potable: Optional[float] = None

def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Pipeline file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_pipeline()
    yield

app = FastAPI(title="Water Potability API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    model = getattr(app.state, "model", None)
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(sample: WaterSample):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Pipeline is not loaded")

    payload = sample.model_dump()
    input_df = pd.DataFrame(
        [[payload[col] for col in RAW_INPUT_COLUMNS]],
        columns=RAW_INPUT_COLUMNS
    )

    prediction = int(model.predict(input_df)[0])

    prob_potable = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        prob_potable = float(probs[1]) if len(probs) > 1 else float(probs[0])

    return PredictionResponse(
        potable=prediction,
        label="potable" if prediction == 1 else "not_potable",
        probability_potable=prob_potable
    )