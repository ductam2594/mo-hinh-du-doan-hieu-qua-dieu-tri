# Import các thư viện cần thiết
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when, isnull
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F

# Tạo Spark Session
spark = SparkSession.builder \
    .appName("GlobalHealthAnalysis") \
    .master("local[*]") \
    .getOrCreate()

# --- 1. Đọc dữ liệu ---
data_path = "D:/nam_3/ky_2/du_lieu_lon/baitaplon/Global Health Statistics.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# In schema của dữ liệu
print("Schema của dữ liệu:")
df.printSchema()

# --- 2. Xử lý giá trị thiếu ---
numeric_cols = [
    "Prevalence Rate (%)", "Incidence Rate (%)", "Mortality Rate (%)", "Recovery Rate (%)",
    "Doctors per 1000", "Hospital Beds per 1000", "Population Affected",
    "Per Capita Income (USD)", "Education Index", "Urbanization Rate (%)",
    "Healthcare Access (%)", "Average Treatment Cost (USD)", "DALYs",
    "Improvement in 5 Years (%)"
]

# Kiểm tra các cột tồn tại trong dataset
existing_cols = [c for c in numeric_cols if c in df.columns]
missing_cols = [c for c in numeric_cols if c not in df.columns]
if missing_cols:
    print("\nCác cột không tồn tại trong dataset:", missing_cols)
numeric_cols = existing_cols

# Điền giá trị thiếu bằng trung bình cho các cột số
for c in numeric_cols:
    mean_val = df.select(mean(c).alias("mean_val")).collect()[0]["mean_val"]
    df = df.withColumn(c, when(isnull(col(c)), mean_val).otherwise(col(c)))

# --- 3. Xử lý ngoại lệ (Outlier Detection) ---
def remove_outliers_iqr(df, col_name):
    quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
    if len(quantiles) < 2:
        return df
    Q1, Q3 = quantiles
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df.filter((col(col_name) >= lower_bound) & (col(col_name) <= upper_bound))

for c in numeric_cols:
    df = remove_outliers_iqr(df, c)

# --- 4. Phân tích khám phá (EDA) ---
# Xu hướng tỷ lệ mắc bệnh trung bình toàn cầu theo thời gian
global_trend = df.groupBy("Year").agg(F.mean("Prevalence Rate (%)").alias("Prevalence Rate Average")).toPandas()

plt.figure(figsize=(10, 5))
sns.lineplot(data=global_trend, x="Year", y="Prevalence Rate Average")
plt.title("Xu hướng tỷ lệ mắc bệnh trung bình toàn cầu theo thời gian")
plt.xlabel("Năm")
plt.ylabel("Tỷ lệ mắc bệnh trung bình (%)")
plt.show()

# Tỷ lệ tử vong theo quốc gia (Top 10)
mortality_by_country = df.groupBy("Country") \
    .agg(F.mean("Mortality Rate (%)").alias("Mortality Rate Average")) \
    .orderBy(F.desc("Mortality Rate Average")) \
    .limit(10) \
    .toPandas()

plt.figure(figsize=(10, 5))
sns.barplot(data=mortality_by_country, x="Country", y="Mortality Rate Average", order=mortality_by_country["Country"])
plt.xticks(rotation=45)
plt.title("Top 10 quốc gia có tỷ lệ tử vong cao nhất")
plt.xlabel("Quốc gia")
plt.ylabel("Tỷ lệ tử vong trung bình (%)")
plt.show()

# --- 5. Feature Engineering ---
# Tạo đặc trưng tương tác (Doctors per 1000 * Hospital Beds per 1000)
if "Doctors per 1000" in numeric_cols and "Hospital Beds per 1000" in numeric_cols:
    df = df.withColumn("Doctors_Beds_Interact", col("Doctors per 1000") * col("Hospital Beds per 1000"))
    numeric_cols.append("Doctors_Beds_Interact")
    print("Đã tạo đặc trưng tương tác: Doctors_Beds_Interact")

# --- 6. Chuẩn hóa dữ liệu ---
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
df = assembler.transform(df)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# --- 7. Chia tập dữ liệu ---
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# --- 8. Huấn luyện và đánh giá mô hình ---
# Định nghĩa hàm đánh giá
def evaluate_model(predictions, label_col):
    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae")
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    print(f"RMSE cho {label_col}: {rmse:.4f}")
    print(f"MAE cho {label_col}: {mae:.4f}")
    return rmse, mae

# Hồi quy tuyến tính cho Mortality Rate (%)
lr = LinearRegression(featuresCol="scaled_features", labelCol="Mortality Rate (%)")
lr_model = lr.fit(train_data)
lr_predictions = lr_model.transform(test_data)

print("\nKết quả mô hình Linear Regression (Mortality Rate):")
lr_rmse, lr_mae = evaluate_model(lr_predictions, "Mortality Rate (%)")

# Vẽ biểu đồ so sánh thực tế và dự đoán
lr_pred_pd = lr_predictions.select("Mortality Rate (%)", "prediction").toPandas()
plt.figure(figsize=(10, 5))
sns.scatterplot(data=lr_pred_pd, x="Mortality Rate (%)", y="prediction", alpha=0.5)
plt.plot([lr_pred_pd["Mortality Rate (%)"].min(), lr_pred_pd["Mortality Rate (%)"].max()],
         [lr_pred_pd["Mortality Rate (%)"].min(), lr_pred_pd["Mortality Rate (%)"].max()], 
         color="red", linestyle="--")
plt.title("Dự đoán tỷ lệ tử vong (Linear Regression)")
plt.xlabel("Thực tế (%)")
plt.ylabel("Dự đoán (%)")
plt.show()

# Random Forest cho Recovery Rate (%)
rf = RandomForestRegressor(featuresCol="scaled_features", labelCol="Recovery Rate (%)")
rf_model = rf.fit(train_data)
rf_predictions = rf_model.transform(test_data)

print("\nKết quả mô hình Random Forest (Recovery Rate):")
rf_rmse, rf_mae = evaluate_model(rf_predictions, "Recovery Rate (%)")

# Vẽ biểu đồ so sánh thực tế và dự đoán
rf_pred_pd = rf_predictions.select("Recovery Rate (%)", "prediction").toPandas()
plt.figure(figsize=(10, 5))
sns.scatterplot(data=rf_pred_pd, x="Recovery Rate (%)", y="prediction", alpha=0.5)
plt.plot([rf_pred_pd["Recovery Rate (%)"].min(), rf_pred_pd["Recovery Rate (%)"].max()],
         [rf_pred_pd["Recovery Rate (%)"].min(), rf_pred_pd["Recovery Rate (%)"].max()], 
         color="red", linestyle="--")
plt.title("Dự đoán tỷ lệ hồi phục (Random Forest)")
plt.xlabel("Thực tế (%)")
plt.ylabel("Dự đoán (%)")
plt.show()

# --- 9. Dừng Spark Session ---
spark.stop()