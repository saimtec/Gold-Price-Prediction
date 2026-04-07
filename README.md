# Gold Price Prediction Project

This project predicts the daily GLD price using market features.

It includes:
- A model training script
- A FastAPI backend for predictions
- A Streamlit web app for user input and results

## What This Project Does

The model learns from historical data and predicts GLD price from these inputs:
- SPX
- USO
- SLV
- EUR/USD

## Project Structure

- `src/train_model.py` trains the model and saves files
- `src/gold_model.py` contains shared ML logic (load data, train, save, load, predict)
- `src/api.py` provides REST API endpoints
- `src/app.py` provides Streamlit UI
- `data/gld_price_data.csv` is the dataset
- `artifacts/` stores generated model and metrics

## Easy Flow

1. Run training once.
2. Training creates model and metrics files in `artifacts/`.
3. FastAPI and Streamlit both load the saved model.
4. User enters feature values.
5. App/API returns predicted GLD price.

## Output Files After Training

When training finishes, these files are created:
- `artifacts/gold_price_model.joblib` (trained model)
- `artifacts/metrics.json` (R2, MAE, RMSE)

## Setup

Use your virtual environment and install dependencies:

```powershell
pip install -r requirements.txt
```

## Run Commands

Train model:

```powershell
python src/train_model.py
```

Run FastAPI server:

```powershell
python -m uvicorn src.api:app --reload
```

Run Streamlit app:

```powershell
python -m streamlit run src/app.py
```

## API Usage

Base URL after running API:
- `http://127.0.0.1:8000`

Health route:
- `GET /`

Prediction route:
- `POST /predict`

Example request body:

```json
{
  "spx": 1447.16,
  "uso": 78.47,
  "slv": 15.18,
  "eurUsd": 1.471692
}
```

Example response:

```json
{
  "predicted_gld_price": 84.98
}
```

## Notes

- If prediction fails with model not found, run `python src/train_model.py` first.
- This project is for learning and experimentation.
- It is not financial advice.

## Get This Repository

Clone from GitHub:

```powershell
git clone https://github.com/saimtec/Gold-Price-Prediction.git
cd Gold-Price-Prediction
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

```

Connect remote and push to main branch:

```powershell
git remote add origin https://github.com/saimtec/Gold-Price-Prediction.git
git branch -M main
git push -u origin main
```
