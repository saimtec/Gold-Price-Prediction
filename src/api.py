from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from .gold_model import load_model, predict_price, FEATURE_COLUMNS
except ImportError:
    from gold_model import load_model, predict_price, FEATURE_COLUMNS


app = FastAPI(title="Gold Price Prediction API", version="1.0.0")


class GoldFeatures(BaseModel):
    spx: float = Field(..., description="S&P 500 index value")
    uso: float = Field(..., description="USO ETF value")
    slv: float = Field(..., description="SLV ETF value")
    eur_usd: float = Field(..., alias="eurUsd", description="EUR/USD exchange rate")


@app.get("/")
def root():
    return {
        "message": "Gold Price Prediction API is running.",
        "required_features": FEATURE_COLUMNS,
    }


@app.post("/predict")
def predict(payload: GoldFeatures):
    model = load_model()
    feature_values = {
        "SPX": payload.spx,
        "USO": payload.uso,
        "SLV": payload.slv,
        "EUR/USD": payload.eur_usd,
    }
    prediction = predict_price(model, feature_values)
    return {"predicted_gld_price": prediction}
