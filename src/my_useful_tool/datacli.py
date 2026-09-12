import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

warnings.filterwarnings("ignore")


class HighAccuracyMLPrep:
    """Production-ready automated data preparation engine optimized for high-throughput memory efficiency and feature selection."""

    def __init__(self, task_type: str = "classification", max_features: int = 40):
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type must be either 'classification' or 'regression'")
        
        self.task_type = task_type
        self.max_features = max_features

    def fit_transform(
        self, file_path: str, target_column: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Ingests raw CSV data, executes vectorized cleaning, feature engineering, signal pruning, and hyperparameter profiling."""
        # 1. Ingest Data with C-engine chunking
        df = pd.read_csv(file_path, engine="c")
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in CSV.")

        # Separate Target and Predictors
        df_target = df[target_column].copy()
        df_features = df.drop(columns=[target_column]).copy()

        # 2. Process Target Variable
        y_final, task_detected = self._process_target(df_target)

        # 3. Vectorized Structural Type Detection & Cleaning
        df_cleaned = self._detect_and_clean_features(df_features)

        # 4. Advanced High-Speed Feature Engineering
        df_engineered = self._engineer_features(df_cleaned)

        # 5. Missing Value Imputation & Standard Scaler with Float32 Downcasting
        X_array, feature_names = self._impute_and_scale(df_engineered)

        # 6. Parallel Signal Pruning via Gradient Boosted Engine
        X_optimized, selected_features = self._prune_features(
            X_array, y_final, feature_names, task_detected
        )

        # 7. Dynamic Hyperparameter Profiling
        recommended_params = self._profile_hyperparameters(
            n_samples=X_optimized.shape[0],
            n_features=X_optimized.shape[1],
            task_detected=task_detected,
        )

        pipeline_diagnostics = {
            "selected_features": selected_features,
            "recommended_hyperparameters": recommended_params,
            "task_detected": task_detected,
        }

        return X_optimized, y_final, pipeline_diagnostics

    def _process_target(self, s: pd.Series) -> Tuple[np.ndarray, str]:
        s = s.dropna()
        if self.task_type == "classification":
            le = LabelEncoder()
            y_encoded = le.fit_transform(s.astype(str))
            n_classes = len(np.unique(y_encoded))
            task_detected = "binary" if n_classes == 2 else "multiclass"
            return y_encoded.astype(np.int32), task_detected
        else:
            y_numeric = pd.to_numeric(s, errors="coerce").fillna(s.median())
            return y_numeric.values.astype(np.float32), "regression"

    def _detect_and_clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_dict = {}

        for col in df.columns:
            series = df[col]

            # Fast Date Parsing via Sampling
            if series.dtype == "object":
                sample = series.dropna().astype(str).head(30)
                if not sample.empty:
                    parsed_dates = pd.to_datetime(series, errors="coerce", format="mixed")
                    if parsed_dates.notna().sum() / len(series) > 0.6:
                        cleaned_dict[f"{col}_year"] = parsed_dates.dt.year.astype("float32")
                        cleaned_dict[f"{col}_month"] = parsed_dates.dt.month.astype("float32")
                        cleaned_dict[f"{col}_day"] = parsed_dates.dt.day.astype("float32")
                        cleaned_dict[f"{col}_dayofweek"] = parsed_dates.dt.dayofweek.astype("float32")
                        continue

            # Numeric Conversion & Downcasting
            numeric_series = pd.to_numeric(series, errors="coerce")
            if numeric_series.notna().sum() / len(series) > 0.5:
                cleaned_dict[f"{col}_clean"] = numeric_series.astype("float32")
            else:
                # Categorical Frequency & Label Encoding
                str_series = series.astype(str).fillna("missing")
                cardinality = str_series.nunique()
                if cardinality <= 50:
                    le = LabelEncoder()
                    cleaned_dict[f"{col}_encoded"] = le.fit_transform(str_series).astype("float32")
                else:
                    freq = str_series.value_counts(normalize=True).to_dict()
                    cleaned_dict[f"{col}_freq"] = str_series.map(freq).astype("float32")

        return pd.DataFrame(cleaned_dict, index=df.index)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        engineered = df.copy()
        numeric_cols = [c for c in df.columns if df[c].dtype in (np.float32, np.float64, "float32", "float64")]

        # Vectorized Outlier Tagging (IQR Boundary)
        if numeric_cols:
            subset_cols = numeric_cols[:10]
            data_matrix = engineered[subset_cols].values
            q1 = np.nanpercentile(data_matrix, 25, axis=0)
            q3 = np.nanpercentile(data_matrix, 75, axis=0)
            iqr = q3 - q1

            valid_mask = iqr > 0
            for idx, col in enumerate(subset_cols):
                if valid_mask[idx]:
                    lower, upper = q1[idx] - 1.5 * iqr[idx], q3[idx] + 1.5 * iqr[idx]
                    col_vals = engineered[col].values
                    engineered[f"{col}_is_outlier"] = ((col_vals < lower) | (col_vals > upper)).astype("float32")

        # Multi-variable interaction terms
        if len(numeric_cols) >= 2:
            pair_count = 0
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    if pair_count >= 10:
                        break
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    engineered[f"{col1}_x_{col2}"] = (
                        engineered[col1].fillna(0) * engineered[col2].fillna(0)
                    ).astype("float32")
                    pair_count += 1
                if pair_count >= 10:
                    break

        return engineered

    def _impute_and_scale(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        df_imputed = df.fillna(df.median(numeric_only=True)).fillna(0)
        feature_names = list(df_imputed.columns)
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df_imputed.values).astype(np.float32)
        return scaled_array, feature_names

    def _prune_features(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str], task_detected: str
    ) -> Tuple[np.ndarray, List[str]]:
        n_features = X.shape[1]
        target_k = min(self.max_features, n_features)

        if HAS_LIGHTGBM:
            if task_detected == "regression":
                model = lgb.LGBMRegressor(n_estimators=30, n_jobs=-1, random_state=42, verbose=-1)
            else:
                model = lgb.LGBMClassifier(n_estimators=30, n_jobs=-1, random_state=42, verbose=-1)
        else:
            if task_detected == "regression":
                model = GradientBoostingRegressor(n_estimators=30, random_state=42)
            else:
                model = GradientBoostingClassifier(n_estimators=30, random_state=42)

        model.fit(X, y)
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:target_k]
        
        return X[:, top_indices], [feature_names[i] for i in top_indices]

    def _profile_hyperparameters(
        self, n_samples: int, n_features: int, task_detected: str
    ) -> Dict[str, Any]:
        base_lr = max(0.01, min(0.1, round(1.0 / np.sqrt(n_samples / 100), 4)))
        num_leaves = max(15, min(127, int(2 ** min(7, np.log2(n_features + 1)))))
        max_depth = max(3, min(10, int(np.log2(num_leaves)) + 1))
        min_child = max(10, min(100, int(n_samples * 0.01)))
        subsample = 0.8 if n_samples > 5000 else 1.0
        colsample = 0.8 if n_features > 15 else 1.0

        return {
            "learning_rate": base_lr,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child,
            "subsample": subsample,
            "colsample_bytree": colsample,
        }