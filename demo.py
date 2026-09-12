import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

from my_useful_tool import HighAccuracyMLPrep

# ---------------------------------------------------------
# 1. Generate 10,000-Row Messy Benchmark Dataset
# ---------------------------------------------------------
print("Generating 10,000-row messy benchmark dataset...")
np.random.seed(42)
n_rows = 10_000

# Continuous numeric with missing values and extreme outliers
account_balance = np.random.exponential(scale=5000, size=n_rows)
account_balance[np.random.choice(n_rows, 1500, replace=False)] = np.nan
account_balance[np.random.choice(n_rows, 200, replace=False)] *= 50  # Outliers

monthly_spend = np.random.normal(loc=300, scale=100, size=n_rows)
monthly_spend[np.random.choice(n_rows, 1000, replace=False)] = np.nan

# Categorical data (Low & High Cardinality)
cities = np.random.choice(["NYC", "London", "Tokyo", "Paris", "Berlin", np.nan], size=n_rows, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
user_ids = [f"USR_{i}" for i in range(n_rows)]  # Very high cardinality

# Timestamps with bad/corrupted entries
dates = pd.date_range(start="2021-01-01", periods=n_rows, freq="h").astype(str).to_numpy()
dates[np.random.choice(n_rows, 800, replace=False)] = "INVALID_DATE_FORMAT"
dates[np.random.choice(n_rows, 500, replace=False)] = np.nan

# Target variable (Non-linear relationship with features + noise)
signal = (
    (np.nan_to_num(account_balance) > 4000).astype(int) * 2.0
    + (np.nan_to_num(monthly_spend) > 350).astype(int) * 1.5
    + np.random.normal(0, 1, size=n_rows)
)
churn_target = np.where(signal > 2.0, "Yes", "No")

# Build and export CSV
benchmark_df = pd.DataFrame({
    "User_ID": user_ids,
    "Account_Balance": account_balance,
    "Monthly_Spend": monthly_spend,
    "Primary_City": cities,
    "Signup_Date": dates,
    "Churned": churn_target
})

file_path = "benchmark_messy_data_10k.csv"
benchmark_df.to_csv(file_path, index=False)
print(f"Dataset created: {file_path} ({os.path.getsize(file_path) / 1024:.2f} KB)")

# ---------------------------------------------------------
# 2. Run HighAccuracyMLPrep Pipeline
# ---------------------------------------------------------
print("\nExecuting HighAccuracyMLPrep engine...")
start_time = time.time()

processor = HighAccuracyMLPrep(task_type="classification", max_features=20)
X, y, diagnostics = processor.fit_transform(
    file_path=file_path,
    target_column="Churned"
)

elapsed_time = time.time() - start_time

print(f"\n--- Pipeline Diagnostics ---")
print(f"Total Processing Time : {elapsed_time:.3f} seconds")
print(f"Processed Matrix Shape: {X.shape}")
print(f"Target Vector Shape   : {y.shape}")
print(f"Selected Features     : {diagnostics['selected_features']}")
print(f"Recommended Tuning    : {diagnostics['recommended_hyperparameters']}")

# ---------------------------------------------------------
# 3. Model Training & Accuracy Validation
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = diagnostics["recommended_hyperparameters"]
params["objective"] = "binary:logistic"
params["eval_metric"] = "logloss"

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n--- Validation Performance ---")
print(f"XGBoost Test Accuracy : {accuracy * 100:.2f}%")
print(f"XGBoost ROC-AUC Score : {roc_auc:.4f}")

# Cleanup benchmark file
if os.path.exists(file_path):
    os.remove(file_path)