import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gradio as gr
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from my_useful_tool import HighAccuracyMLPrep

def process_and_train(file_obj, target_column, max_features):
    if file_obj is None:
        return "Please upload a CSV file.", None, ""
    
    file_path = file_obj.name
    
    # 1. Run Data Prep Engine
    processor = HighAccuracyMLPrep(task_type="classification", max_features=int(max_features))
    X, y, diagnostics = processor.fit_transform(file_path=file_path, target_column=target_column)
    
    # 2. Train Validation XGBoost Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = diagnostics.get("recommended_hyperparameters", {})
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
        auc_str = f"{auc:.4f}"
    except Exception:
        auc_str = "N/A"
    
    # 3. Output Processed Data
    processed_df = pd.DataFrame(X)
    processed_df["target"] = y
    output_path = "processed_dataset.csv"
    processed_df.to_csv(output_path, index=False)
    
    report = f"""
    ### 🚀 Processing Complete!
    - **Original Shape:** {pd.read_csv(file_path).shape}
    - **Processed Matrix:** {X.shape}
    - **Selected Features:** {len(diagnostics.get('selected_features', []))}
    
    ### 📊 Validation Results
    - **Accuracy:** {acc * 100:.2f}%
    - **ROC-AUC Score:** {auc_str}
    
    ### ⚙️ Recommended Hyperparameters
    ```json
    {diagnostics.get('recommended_hyperparameters', {})}
    ```
    """
    
    return report, output_path

# Gradio Interface Setup
with gr.Blocks(title="HighAccuracyMLPrep") as demo:
    gr.Markdown("# ⚡ HighAccuracyMLPrep Engine")
    gr.Markdown("Upload a raw CSV dataset to automatically clean, handle missing values, prune features, and extract model hyperparameters.")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Raw CSV Dataset", file_types=[".csv"])
            target_col = gr.Textbox(label="Target Column Name", value="Churned")
            max_feats = gr.Slider(minimum=5, maximum=50, value=20, step=1, label="Max Features")
            submit_btn = gr.Button("Process Dataset & Train Model", variant="primary")
            
        with gr.Column():
            status_output = gr.Markdown(label="Diagnostics Report")
            file_output = gr.File(label="Download Cleaned CSV Matrix")
            
    submit_btn.click(
        fn=process_and_train,
        inputs=[file_input, target_col, max_feats],
        outputs=[status_output, file_output]
    )

if __name__ == "__main__":
    demo.launch()