import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import sqrt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor

# ==========================================================
# STEP 1: LOAD DATA
# ==========================================================

print("\nLoading dataset...")

DATASET_PATH = "household_power_consumption_500rows.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print("Loaded:", df.shape)

# ==========================================================
# STEP 2: DATETIME PARSING
# ==========================================================

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce"
)

df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

df = df[["Datetime", "Global_active_power"]].dropna()

# IMPORTANT: sort index
df = df.sort_values("Datetime")

# ==========================================================
# STEP 3: TIME SERIES (FIXED - NO OVER-AGGRESSIVE RESAMPLE)
# ==========================================================

df = df.set_index("Datetime")

# Keep original resolution (DO NOT resample to hourly for small dataset)
ts = df["Global_active_power"].interpolate()

print("Time series length:", len(ts))

# ==========================================================
# STEP 4: FEATURE ENGINEERING
# ==========================================================

data = ts.reset_index()
data.columns = ["Datetime", "Usage"]

data["hour"] = data["Datetime"].dt.hour
data["day"] = data["Datetime"].dt.day
data["month"] = data["Datetime"].dt.month
data["weekday"] = data["Datetime"].dt.weekday
data["weekend"] = (data["weekday"] >= 5).astype(int)

# Lag features
data["lag1"] = data["Usage"].shift(1)
data["lag24"] = data["Usage"].shift(24)

data = data.dropna().reset_index(drop=True)

# ==========================================================
# STEP 5: TRAIN TEST SPLIT
# ==========================================================

split_index = int(len(data) * 0.8)

train = data.iloc[:split_index]
test = data.iloc[split_index:]

print("\nTraining Samples:", len(train))
print("Testing Samples :", len(test))

# ==========================================================
# STEP 6: ARIMA
# ==========================================================

train_ts = ts.iloc[:split_index]
test_ts = ts.iloc[split_index:]

arima_model = ARIMA(train_ts, order=(5, 1, 0))
arima_fit = arima_model.fit()

arima_pred = arima_fit.forecast(steps=len(test_ts))

# ==========================================================
# STEP 7: PROPHET
# ==========================================================

prophet_df = ts.reset_index()
prophet_df.columns = ["ds", "y"]

split_prophet = int(len(prophet_df) * 0.8)

train_prophet = prophet_df.iloc[:split_prophet]

model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True
)

model.fit(train_prophet)

future = model.make_future_dataframe(
    periods=len(prophet_df) - split_prophet,
    freq="H"
)

forecast = model.predict(future)

prophet_pred = forecast["yhat"].iloc[split_prophet:].values
prophet_actual = prophet_df["y"].iloc[split_prophet:].values

# ==========================================================
# STEP 8: XGBOOST
# ==========================================================

features = ["hour", "day", "month", "weekday", "weekend", "lag1", "lag24"]

X_train = train[features]
y_train = train["Usage"]

X_test = test[features]
y_test = test["Usage"]

xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# ==========================================================
# STEP 9: EVALUATION
# ==========================================================

def evaluate(y_true, y_pred, name):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{name}")
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))

evaluate(test_ts, arima_pred, "ARIMA")
evaluate(prophet_actual, prophet_pred, "PROPHET")
evaluate(y_test, xgb_pred, "XGBOOST")

# ==========================================================
# STEP 10: PLOTS
# ==========================================================

plt.figure(figsize=(12,5))
plt.plot(test_ts.values, label="Actual")
plt.plot(arima_pred.values, label="ARIMA")
plt.title("ARIMA Forecast")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(prophet_actual, label="Actual")
plt.plot(prophet_pred, label="Prophet")
plt.title("Prophet Forecast")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(y_test.values, label="Actual")
plt.plot(xgb_pred, label="XGBoost")
plt.title("XGBoost Prediction")
plt.legend()
plt.grid()
plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY 🚀")