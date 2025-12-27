"""
Train XGBoost model on cleaned retail dataset and save to model.bin
Run:
    python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

# ===============================================================
# 1. Load dữ liệu từ file CSV
# ===============================================================
DATA_PATH = "data_cleaned (1).csv"   # đổi tên theo file của bạn

df = pd.read_csv(DATA_PATH)

print(">>> Loaded dataset:", df.shape)
print(df.head())

# ===============================================================
# 2. Xác định cột target + chọn feature
# ===============================================================
TARGET = "Weekly_Sales"

if TARGET not in df.columns:
    raise ValueError(f"Không tìm thấy cột target: {TARGET}")

# Chọn toàn bộ cột numeric làm feature
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Bỏ cột target khỏi feature list
feature_cols = [c for c in numeric_cols if c != TARGET]

print("\n>>> Features used:")
print(feature_cols)

X = df[feature_cols]
y = df[TARGET].values

# ===============================================================
# 3. Train/Test Split
# ===============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n>>> Train/Test sizes:", X_train.shape, X_test.shape)

# ===============================================================
# 4. Train XGBoost Regressor
# ===============================================================
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

# ===============================================================
# 5. Đánh giá mô hình
# ===============================================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

# ===============================================================
# 6. Lưu model dạng .bin giống file mẫu
# ===============================================================
model.save_model("model.bin")
print("\n>>> Saved model to model.bin")

# ===============================================================
# 7. Lưu feature order (quan trọng cho Flask API)
# ===============================================================
joblib.dump(feature_cols, "feature_columns.pkl")
print(">>> Saved feature order to feature_columns.pkl")
