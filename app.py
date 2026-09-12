import os
import joblib
import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier, LGBMRegressor

def inspect_csv(file):
    """Parses CSV upon upload and populates target column dropdown."""
    if file is None:
        return gr.update(choices=[], value=None)
    df = pd.read_csv(file.name)
    cols = df.columns.tolist()
    choices = ["(None - Unsupervised Clustering)"] + cols
    return gr.update(choices=choices, value=choices[1] if len(choices) > 1 else None)

def process_data_and_train(file, target_col, max_features):
    if file is None:
        return "Please upload a CSV file.", None, None

    df = pd.read_csv(file.name)
    
    # Separate features and target
    if target_col and target_col != "(None - Unsupervised Clustering)":
        if target_col not in df.columns:
            return f"Error: Column '{target_col}' not found.", None, None
        X = df.drop(columns=[target_col])
        y = df[target_col]
        is_supervised = True
    else:
        X = df.copy()
        y = None
        is_supervised = False

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Handle missing values
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        X[col] = X[col].fillna("Missing")

    # One-Hot Encoding & Scaling
    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )
    
    X_processed = ct.fit_transform(X)
    
    # Truncate to Max Features
    if X_processed.shape[1] > max_features:
        X_processed = X_processed[:, :max_features]

    # Save cleaned matrix CSV
    processed_df = pd.DataFrame(X_processed)
    if is_supervised:
        processed_df["target"] = y.values
    processed_df.to_csv("processed_dataset.csv", index=False)

    # Supervised Learning Mode
    if is_supervised:
        is_classification = y.nunique() <= 20 or y.dtype == 'object'
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        
        if is_classification:
            model = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            metrics_summary = f"🚀 Supervised Training Complete!\n• Task: Classification\n• Accuracy: {score:.4f}"
        else:
            model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            metrics_summary = f"🚀 Supervised Training Complete!\n• Task: Regression\n• R² Score: {score:.4f}"

        joblib.dump(model, "trained_model.joblib")
        best_params = model.get_params()

    # Unsupervised Clustering Mode
    else:
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_processed)
        processed_df["cluster_label"] = clusters
        processed_df.to_csv("processed_dataset.csv", index=False)

        joblib.dump(kmeans, "trained_model.joblib")
        metrics_summary = f"⚡ Unsupervised Clustering Complete!\n• Algorithm: K-Means (k=3)\n• Inertia: {kmeans.inertia_:.2f}"
        best_params = kmeans.get_params()

    status_text = f"{metrics_summary}\n\n⚙️ Model Parameters:\n{best_params}"
    return status_text, "processed_dataset.csv", "trained_model.joblib"

# UI Scaffold
with gr.Blocks(title="HighAccuracyMLPrep Engine") as demo:
    gr.Markdown("# ⚡ HighAccuracyMLPrep Engine")
    gr.Markdown("Upload a raw CSV dataset to clean missing values, encode features, train models, or perform unsupervised clustering.")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Raw CSV Dataset", file_types=[".csv"])
            target_dropdown = gr.Dropdown(label="Target Column Name", choices=[], value=None, interactive=True)
            max_feat_slider = gr.Slider(minimum=5, maximum=50, value=20, step=1, label="Max Features")
            btn = gr.Button("Process Dataset & Train Model", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Execution Summary & Metrics", lines=8)
            processed_file_output = gr.File(label="Download Processed Dataset (CSV)")
            model_artifact_output = gr.File(label="Download Trained Model (.joblib)")

    file_input.change(fn=inspect_csv, inputs=[file_input], outputs=[target_dropdown])
    btn.click(
        fn=process_data_and_train,
        inputs=[file_input, target_dropdown, max_feat_slider],
        outputs=[output_text, processed_file_output, model_artifact_output]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
