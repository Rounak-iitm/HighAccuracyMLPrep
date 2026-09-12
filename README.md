---
title: HighAccuracyMLPrep Engine
author: Rounak
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
---

# ⚡ HighAccuracyMLPrep Engine

A production-grade, cloud-deployed machine learning pipeline designed to bridge the gap between messy, raw, real-world tabular data and high-accuracy predictive models.

Upload a raw CSV file, dynamically select your target column, and let the engine automate data cleaning, feature vectorization, model training, and artifact export in seconds.

---

## 🚀 Core Features

- **Dynamic Header Parsing:** Automatically populates the target column selection dropdown upon CSV upload to eliminate capitalization and spelling errors.
- **Automated Data Sanitization:** Handles missing data by filling numerical nulls with column medians and categorical nulls with explicit string indicators.
- **Leakage-Free Vectorization:** Encodes categorical variables via `OneHotEncoder` and normalizes numerical features using `StandardScaler`.
- **Dimensionality Control:** Features an interactive slider (5 to 50 features) to prune noisy feature vectors and prevent overfitting.
- **Supervised Machine Learning:** Automatically detects classification vs. regression tasks based on target label cardinality and trains tabular gradient boosting trees via LightGBM.
- **Unsupervised Clustering Fallback:** Executes automated K-Means clustering ($k=3$) when no target outcome column is specified.
- **Dual Artifact Generation:** Exports both the preprocessed feature matrix (`processed_dataset.csv`) and the binary model object (`trained_model.joblib`) for immediate downstream deployment.

---

## 🛠️ Tech Stack & Dependencies

- **Framework & UI:** Gradio `6.27.0`
- **Core ML Libraries:** LightGBM, Scikit-Learn, Pandas, NumPy, Joblib
- **Runtime Environment:** Python `3.11.9` on Render Cloud Web Services

---

## 📊 How to Use

1. **Upload Dataset:** Drop any raw `.csv` file into the file upload box.
2. **Select Target Column:** Choose your outcome column from the dynamic dropdown menu (or select `(None - Unsupervised Clustering)` for unlabeled data).
3. **Adjust Max Features:** Use the slider to set your desired maximum feature limit.
4. **Execute Pipeline:** Click **Process Dataset & Train Model**.
5. **Download Artifacts:** View performance metrics (`Accuracy`, `R² Score`, or `Inertia`) and download both `processed_dataset.csv` and `trained_model.joblib`.
