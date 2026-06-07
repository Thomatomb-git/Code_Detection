import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack
from datasets import load_dataset
import matplotlib.pyplot as plt

# Resolve absolute paths dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "saved_models", "ml2")
metrics_dir = os.path.join(script_dir, "metrics")

# 1. Load dataset
print("Loading dataset...")
datasets = load_dataset("basakdemirok/AIGCodeSet")
df = datasets['train'].to_pandas()
df_clean = df.copy()
df_clean['code'] = df_clean['code'].fillna('')

# 2. Load model, vectorizer, and scaler
print("Loading model components...")
model = joblib.load(os.path.join(model_dir, "model.joblib"))
vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))

# 3. Preprocessing features
print("Processing features...")
X_tfidf = vectorizer.transform(df_clean['code'])
meta_cols = ['lines', 'code_lines', 'comments', 'functions', 'blank_lines']
X_meta = df_clean[meta_cols].values
X_meta_scaled = scaler.transform(X_meta)
X_combined = hstack([X_tfidf, X_meta_scaled])
y = df_clean['label'].values

# 4. Train-Test Split (menggunakan random_state yang sama dengan ml2.py)
_, X_test, _, y_test = train_test_split(
    X_combined, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y 
)

# 5. Predict
print("Predicting on test set...")
y_pred = model.predict(X_test)

# 6. Print Classification Report
print("\n=== HASIL EVALUASI MODEL (TF-IDF + Logistic) ===")
report = classification_report(y_test, y_pred, target_names=["Human (0)", "AI (1)"])
print(report)

# 7. Simpan Laporan Evaluasi ke File Teks
os.makedirs(metrics_dir, exist_ok=True)
with open(os.path.join(metrics_dir, "report_ml2.txt"), "w") as f:
    f.write("=== HASIL EVALUASI MODEL (TF-IDF + Logistic) ===\n")
    f.write(report)

# 8. Save Confusion Matrix PNG
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
plt.title("Confusion Matrix - Logistic Regression + TF-IDF")
output_path = os.path.join(metrics_dir, "confusion_matrix_ml2.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Confusion Matrix disimpan di: {output_path}")
