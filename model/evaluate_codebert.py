import os
import torch
import numpy as np
from datasets import load_dataset, Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Resolve absolute paths dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "saved_models", "codebert")
metrics_dir = os.path.join(script_dir, "metrics")

# 1. Load dataset
print("Loading dataset...")
datasets = load_dataset("basakdemirok/AIGCodeSet")
df = datasets['train'].to_pandas()
df_clean = df[['code', 'label']].copy()
df_clean['code'] = df_clean['code'].fillna('')

hf_dataset = Dataset.from_pandas(df_clean)
hf_dataset = hf_dataset.cast_column("label", ClassLabel(names=["Human", "AI"]))
hf_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
test_dataset = hf_dataset["test"]

# 2. Load Model and Tokenizer
print("Loading CodeBERT model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 3. Batch Inference (for efficiency)
print(f"Evaluating CodeBERT model on {device} (this might take a few minutes)...")
y_test = test_dataset["label"]
y_pred = []

# Process in batches to avoid OOM
batch_size = 16
total_samples = len(test_dataset)

for i in range(0, total_samples, batch_size):
    if (i // batch_size) % 10 == 0:
        print(f"Processing batch {i // batch_size + 1}/{int(np.ceil(total_samples / batch_size))}...")
    batch = test_dataset[i:i+batch_size]
    inputs = tokenizer(
        batch["code"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    y_pred.extend(preds)

y_pred = np.array(y_pred)

# 4. Print Classification Report
print("\n=== HASIL EVALUASI MODEL (CodeBERT) ===")
report = classification_report(y_test, y_pred, target_names=["Human (0)", "AI (1)"])
print(report)

# 5. Simpan Laporan Evaluasi ke File Teks
os.makedirs(metrics_dir, exist_ok=True)
with open(os.path.join(metrics_dir, "report_codebert.txt"), "w") as f:
    f.write("=== HASIL EVALUASI MODEL (CodeBERT) ===\n")
    f.write(report)

# 6. Save Confusion Matrix PNG
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
plt.title("Confusion Matrix - CodeBERT")
output_path = os.path.join(metrics_dir, "confusion_matrix_codebert.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Confusion Matrix disimpan di: {output_path}")
