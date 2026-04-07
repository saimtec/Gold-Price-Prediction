# Gold Price Prediction

This project predicts GLD price using market features.

Architecture:
- FastAPI backend handles model loading and prediction.
- Streamlit frontend sends user input to backend predict API.
- Streamlit does not need local model files in cloud deployment.

## Project Structure

- src/api.py: FastAPI backend endpoints
- src/train_model.py: trains model and writes artifacts
- src/gold_model.py: shared ML logic
- src/app.py: Streamlit frontend (API client)
- app.py: root launcher for Streamlit
- data/gld_price_data.csv: dataset
- artifacts/: generated model and metrics (backend side)
- requirements.txt: Python dependencies

## Local Run

1. Install dependencies
pip install -r requirements.txt

2. Train model once
python src/train_model.py

3. Start backend
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000

4. Start frontend
streamlit run app.py

5. Open frontend
http://localhost:8501

## Deploy Backend on Render

1. Create new Web Service in Render.
2. Connect GitHub repository:
https://github.com/saimtec/Gold-Price-Prediction
3. Use these settings:
- Runtime: Python
- Branch: main
- Build Command:
pip install -r requirements.txt && python src/train_model.py
- Start Command:
uvicorn src.api:app --host 0.0.0.0 --port $PORT
4. Deploy.
5. Verify backend URL:
- GET https://your-render-url.onrender.com/
- Open docs: https://your-render-url.onrender.com/docs

## Deploy Frontend on Streamlit Community Cloud

1. Create a new Streamlit app from this GitHub repo.
2. Set main file path to:
app.py
3. Add a Streamlit secret named BACKEND_URL with your Render URL.

Example secrets value:
BACKEND_URL = "https://your-render-url.onrender.com"

4. Deploy app.

Now frontend sends prediction requests to Render backend.

## API Contract

POST /predict

Request JSON:
{
  "spx": 1447.16,
  "uso": 78.47,
  "slv": 15.18,
  "eurUsd": 1.471692
}

Response JSON:
{
  "predicted_gld_price": 84.98
}

## Git Commands

Clone:
git clone https://github.com/saimtec/Gold-Price-Prediction.git
cd Gold-Price-Prediction

Push updates:
git add .
git commit -m "update"
git push origin main
