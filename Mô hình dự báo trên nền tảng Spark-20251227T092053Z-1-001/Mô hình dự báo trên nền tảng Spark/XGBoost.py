# File: spark_xgboost_retail.py
# Yêu cầu: pyspark, pandas, numpy, scikit-learn, xgboost
# Đường dẫn file (đã upload): /mnt/data/data_cleaned (1).csv

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # để lưu model

# =========================
# 1. Khởi tạo SparkSession
# =========================
spark = SparkSession.builder \
    .appName("RetailSales_XGBoost_Spark") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# =========================
# 2. Đọc dữ liệu
# =========================
input_path = "data_cleaned (1).csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)

# =========================
# 3. Thống kê mô tả
df.describe().show()
# =========================
# =========================
# 4. Tạo lag features & rolling features
#    Giả sử dữ liệu có các cột: Store, Dept, Date, Weekly_Sales
# =========================
target_col = "Weekly_Sales"
if target_col not in df.columns:
    raise ValueError(f"Cột target '{target_col}' không tồn tại trong dữ liệu")

# # Sắp xếp theo cửa hàng, dept, date để tạo lag
# w = Window.partitionBy("Store", "Dept").orderBy("Date")
#
# # lag 1 (doanh thu tuần trước)
# df = df.withColumn("lag_1_sales", F.lag(F.col(target_col), 1).over(w))
#
# # rolling mean 4 tuần (preceding 4 weeks, exclude current)
# w_rows = Window.partitionBy("Store", "Dept").orderBy("Date").rowsBetween(-4, -1)
# df = df.withColumn("rolling_mean_4", F.avg(F.col(target_col)).over(w_rows))
#
# # fill nulls ở lag/rolling bằng 0 hoặc phương án khác
# df = df.fillna({"lag_1_sales": 0.0, "rolling_mean_4": 0.0})

# =========================
# 5. Chọn feature và xử lý categorical
#    - Dùng StringIndexer + OneHotEncoder (Spark ML) nếu cần
#    - Ở đây đơn giản: chuyển Index cho categorical, giữ numeric
# =========================
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

categorical_cols = [c for c, t in df.dtypes if t in ('string',) and c not in ('Date',)]
# exclude target if misdetected
categorical_cols = [c for c in categorical_cols if c != target_col]

indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep") for col in categorical_cols]
# onehot
encoders = [OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_ohe") for col in categorical_cols]

# Chọn feature numeric
numeric_cols = [c for c, t in df.dtypes if t in ('double', 'int', 'bigint', 'float', 'long') and c not in (target_col,)]
# đảm bảo tính nhất quán: bỏ các cột index/ohe trước
numeric_cols = [c for c in numeric_cols if c not in [col + "_idx" for col in categorical_cols] + [col + "_ohe" for col in categorical_cols]]

# feature list = numeric + ohe columns
ohe_cols = [col + "_ohe" for col in categorical_cols]
feature_cols = numeric_cols + ohe_cols

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

stages = []
stages += indexers
stages += encoders
stages.append(assembler)
pipeline = Pipeline(stages=stages)

# Fit pipeline
pipeline_model = pipeline.fit(df)
df_transformed = pipeline_model.transform(df)

# Lấy DataFrame có 2 cột: features (Vector) và label
data_for_training = df_transformed.select("features", F.col(target_col).alias("label"), "Date")

# =========================
# 6. Split train/test theo thời gian
#    (ví dụ: 80% train, 20% test by date cutoff)
# =========================
# tìm ngày cutoff
dates = data_for_training.select(F.min("Date").alias("min_date"), F.max("Date").alias("max_date")).collect()[0]
min_date, max_date = dates["min_date"], dates["max_date"]

# cutoff = min_date + 80% duration
# convert to pandas datetime to compute cutoff
pd_min = pd.to_datetime(min_date)
pd_max = pd.to_datetime(max_date)
cutoff = pd_min + (pd_max - pd_min) * 0.8
cutoff_str = cutoff.strftime("%Y-%m-%d")

train_df = data_for_training.filter(F.col("Date") <= F.lit(cutoff_str)).drop("Date")
test_df  = data_for_training.filter(F.col("Date") >  F.lit(cutoff_str)).drop("Date")

# =========================
# 7. Chuyển sang Pandas / NumPy để train XGBoost (driver)
#    (nếu dữ liệu quá lớn, bạn có thể lưu ra file và train phân mảnh hoặc dùng xgboost4j-spark)
# =========================
# Chuyển Vector -> array bằng toPandas() rồi unpack
train_pd = train_df.toPandas()
test_pd  = test_df.toPandas()

# features đang ở dạng DenseVector, convert sang 2D numpy arrays
X_train = np.vstack(train_pd['features'].apply(lambda v: v.toArray()).values)
y_train = train_pd['label'].values

X_test = np.vstack(test_pd['features'].apply(lambda v: v.toArray()).values)
y_test = test_pd['label'].values

# =========================
# 8. Huấn luyện XGBoost Regressor
# =========================
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)



# =========================
# 9. Dự báo & Đánh giá
# =========================
y_pred = xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test==0, 1e-8, y_test))) * 100
smape = np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred) + 1e-8)) * 100
r2 = r2_score(y_test, y_pred)
n = len(y_test); p = X_train.shape[1]

print("===== XGBoost on Spark-processed data =====")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE (%): {mape:.2f}")
print(f"SMAPE (%): {smape:.2f}")
print(f"R2: {r2:.4f}")

# =========================
# 10. Feature importance (theo XGBoost feature index)
#     Map lại tên feature từ VectorAssembler inputCols
# =========================
feature_names = feature_cols  # từ phía Spark VectorAssembler
importances = xgb_model.feature_importances_
# sort and display
fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print("\nTop features:")
for name, imp in fi[:20]:
    print(f"{name:30} {imp:.4f}")

# =========================

# =========================
# 11. Linear Regression Model
# =========================
from sklearn.linear_model import LinearRegression

# Khởi tạo model Linear Regression
lr_model = LinearRegression()

# Train
lr_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)

# =========================
# 12. Đánh giá mô hình Linear Regression
# =========================
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mape_lr = np.mean(np.abs((y_test - y_pred_lr) / np.where(y_test==0, 1e-8, y_test))) * 100
smape_lr = np.mean(2 * np.abs(y_pred_lr - y_test) / (np.abs(y_test) + np.abs(y_pred_lr) + 1e-8)) * 100
r2_lr = r2_score(y_test, y_pred_lr)

print("\n===== Linear Regression Results =====")
print(f"MAE: {mae_lr:.4f}")
print(f"RMSE: {rmse_lr:.4f}")
print(f"MAPE (%): {mape_lr:.2f}")
print(f"SMAPE (%): {smape_lr:.2f}")
print(f"R2: {r2_lr:.4f}")
# =========================
# 13. So sánh 2 mô hình
# =========================
print("\n===================== SO SÁNH XGBOOST vs LINEAR REGRESSION =====================")
print(f"{'Metric':<10} | {'XGBoost':>12} | {'LinearReg':>12}")
print("-" * 50)
print(f"{'MAE':<10} | {mae:12.4f} | {mae_lr:12.4f}")
print(f"{'RMSE':<10} | {rmse:12.4f} | {rmse_lr:12.4f}")
print(f"{'MAPE%':<10} | {mape:12.2f} | {mape_lr:12.2f}")
print(f"{'SMAPE%':<10} | {smape:12.2f} | {smape_lr:12.2f}")
print(f"{'R2':<10} | {r2:12.4f} | {r2_lr:12.4f}")
# ===============================================================
# 11. TẠO HÀM TÍNH METRICS
# ===============================================================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    smape = np.mean(2 * np.abs(y_pred - y_true) /
                    (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, smape, r2


# ===============================================================
# 12. CHẠY 10 RANDOM STATE CHO LINEAR REGRESSION
# ===============================================================
from sklearn.linear_model import LinearRegression

def run_10_times_linear_regression(X_train, y_train, X_test, y_test):
    all_metrics = []
    for seed in range(10):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        all_metrics.append(evaluate_metrics(y_test, y_pred))
    return np.mean(all_metrics, axis=0)

print("\n===== LINEAR REGRESSION (10 RANDOM STATE AVG) =====")
mae_lr, rmse_lr, mape_lr, smape_lr, r2_lr = run_10_times_linear_regression(
    X_train, y_train, X_test, y_test
)

print(f"MAE:   {mae_lr:.4f}")
print(f"RMSE:  {rmse_lr:.4f}")
print(f"MAPE:  {mape_lr:.2f}")
print(f"SMAPE: {smape_lr:.2f}")
print(f"R2:    {r2_lr:.4f}")


# ===============================================================
# 13. CHẠY 10 RANDOM STATE CHO XGBOOST
# ===============================================================
def run_10_times_xgboost(X_train, y_train, X_test, y_test):
    all_metrics = []
    for seed in range(10):
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=seed,
            n_jobs=-1
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        y_pred = xgb_model.predict(X_test)
        all_metrics.append(evaluate_metrics(y_test, y_pred))
    return np.mean(all_metrics, axis=0)

print("\n===== XGBOOST (10 RANDOM STATE AVG) =====")
mae_xgb, rmse_xgb, mape_xgb, smape_xgb, r2_xgb = run_10_times_xgboost(
    X_train, y_train, X_test, y_test
)

print(f"MAE:   {mae_xgb:.4f}")
print(f"RMSE:  {rmse_xgb:.4f}")
print(f"MAPE:  {mape_xgb:.2f}")
print(f"SMAPE: {smape_xgb:.2f}")
print(f"R2:    {r2_xgb:.4f}")


# ===============================================================
# 14. BẢNG SO SÁNH TRUNG BÌNH 10 LẦN TRAIN
# ===============================================================
print("\n===================== SO SÁNH XGBOOST vs LINEAR REGRESSION (AVG of 10 runs) =====================")
print(f"{'Metric':<10} | {'LinearReg':>12} | {'XGBoost':>12}")
print("-" * 50)
print(f"{'MAE':<10} | {mae_lr:12.4f} | {mae_xgb:12.4f}")
print(f"{'RMSE':<10} | {rmse_lr:12.4f} | {rmse_xgb:12.4f}")
print(f"{'MAPE%':<10} | {mape_lr:12.2f} | {mape_xgb:12.2f}")
print(f"{'SMAPE%':<10} | {smape_lr:12.2f} | {smape_xgb:12.2f}")
print(f"{'R2':<10} | {r2_lr:12.4f} | {r2_xgb:12.4f}")


# ===============================================================
# 15. Lưu model XGBoost cuối cùng
# ===============================================================
joblib.dump(xgb_model, "xgboost_retail_model.pkl")

# ===============================================================
# 16. Kết thúc Spark
# ===============================================================
spark.stop()
# =========================
