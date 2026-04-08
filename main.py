from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "NFLX.csv"
INDEX_FILE = BASE_DIR / "index.html"
TRAINING_WINDOW = 30

app = FastAPI(title="Netflix Stock Prediction Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

full_data = pd.read_csv(DATA_FILE)
training_data = full_data.tail(TRAINING_WINDOW).copy()
training_data["Day"] = np.arange(1, len(training_data) + 1)

X = training_data[["Day"]].values
y = training_data["Close"].values

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

next_day_value = len(training_data) + 1
predicted_price = float(model.predict(poly.transform(np.array([[next_day_value]])))[0])
training_predictions = model.predict(X_poly)
score = float(r2_score(y, training_predictions))

last_row = training_data.iloc[-1]
previous_row = training_data.iloc[-2]
price_change = float(last_row["Close"] - previous_row["Close"])
price_change_percent = float((price_change / previous_row["Close"]) * 100)
average_close = float(training_data["Close"].mean())


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/api/summary")
def get_summary():
    return {
        "company": "Netflix",
        "ticker": "NFLX",
        "model": "Polynomial Regression",
        "degree": 3,
        "training_window": TRAINING_WINDOW,
        "latest_date": str(last_row["Date"]),
        "latest_close": float(last_row["Close"]),
        "previous_close": float(previous_row["Close"]),
        "price_change": price_change,
        "price_change_percent": price_change_percent,
        "average_close": average_close,
        "r2_score": score,
    }


@app.get("/api/predict")
def predict():
    return {
        "predicted_price": predicted_price,
        "prediction_day": next_day_value,
        "prediction_for_date_after": str(last_row["Date"]),
    }


@app.get("/predict")
def predict_legacy():
    return {"predicted_price": predicted_price}
