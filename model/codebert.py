import pandas as pd
import numpy as np
import torch
from datasets import Dataset, load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import evaluate

# ==========================================
# 1. PERSIAPAN DATA
# ==========================================
print("1. Memuat dataset...")
# GANTI NAMA FILE INI dengan file dataset gabungan kamu!
# Asumsi: File CSV ini memiliki kolom 'code' dan 'label' (0 untuk Human, 1 untuk AI)
try:
    datasets = load_dataset("basakdemirok/AIGCodeSet")
    df = datasets['train'].to_pandas()
except FileNotFoundError:
    print(f"Error: File tidak ditemukan. Pastikan lokasinya benar.")
    exit()

# Membersihkan data (ambil kolom yang penting saja dan pastikan tidak ada null)
df_clean = df[['code', 'label']].copy()
df_clean['code'] = df_clean['code'].fillna('')

# Ubah format Pandas menjadi Hugging Face Dataset
print("2. Mengonversi data ke format Hugging Face Dataset...")
hf_dataset = Dataset.from_pandas(df_clean)
hf_dataset = hf_dataset.cast_column("label", ClassLabel(names=["Human", "AI"]))

# Split Dataset menjadi 80% Train dan 20% Test
hf_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")

# ==========================================
# 2. TOKENISASI CODEBERT
# ==========================================
print("3. Memuat Tokenizer CodeBERT dan melakukan tokenisasi data...")
# Ini akan mendownload tokenizer jika belum ada di cache
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def tokenize_function(examples):
    # Truncation: Potong kode jika lebih dari 512 token
    # Padding: Tambahkan token kosong jika kurang dari 512 agar ukurannya seragam
    return tokenizer(examples["code"], padding="max_length", truncation=True, max_length=512)

# Terapkan tokenisasi ke seluruh dataset (batched=True agar lebih cepat)
tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

# ==========================================
# 3. SETUP MODEL & METRIK EVALUASI
# ==========================================
print("4. Memuat Model CodeBERT (bersiaplah, ini mungkin mendownload model ~480MB)...")
# num_labels=2 karena klasifikasi biner: 0 (Human) dan 1 (AI)
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base", 
    num_labels=2,
    use_safetensors=True
)

print("5. Menyiapkan metrik akurasi...")
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# ==========================================
# 4. KONFIGURASI TRAINING (FINE-TUNING)
# ==========================================
print("6. Menyiapkan argumen training...")
training_args = TrainingArguments(
    output_dir="./codebert-temp-checkpoints", # Folder SEMENTARA untuk checkpoint selama training
    eval_strategy="epoch",               # Evaluasi setiap akhir epoch
    save_strategy="epoch",               # Simpan checkpoint setiap akhir epoch
    learning_rate=2e-5,                  # Learning rate standar untuk fine-tuning Transformer
    per_device_train_batch_size=8,       # Batch size untuk GPU. Jika VRAM kepenuhan (OOM), turunkan ke 4 atau 2
    per_device_eval_batch_size=8,
    num_train_epochs=8,                  # siklus training
    weight_decay=0.01,
    load_best_model_at_end=True,         # Gunakan model terbaik di akhir training
    metric_for_best_model="accuracy",    # Patokan model terbaik adalah akurasinya
    fp16=True,                           # (PENTING UNTUK GPU RTX) Mempercepat training & hemat VRAM
    logging_dir='./logs',                # Folder log
    logging_steps=50,
    warmup_ratio=0.1,                    # TRIK BARU: 10% langkah pertama untuk pemanasan
    lr_scheduler_type="cosine",          # TRIK BARU: Cosine decay agar model stabil di akhir
    save_total_limit=1                   # Simpan hanya 1 checkpoint terbaru (hemat disk)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Berhenti jika 2 epoch berturut-turut akurasi tidak naik
)

# ==========================================
# 5. MULAI TRAINING!
# ==========================================
print("7. MEMULAI PROSES TRAINING DI GPU! 🚀 (Silakan tunggu, ini butuh waktu)")
trainer.train()

# ==========================================
# 6. EVALUASI HASIL AKHIR
# ==========================================
print("8. Melakukan evaluasi akhir pada data Test...")
eval_results = trainer.evaluate()
print("\n" + "="*40)
print(f"TRAINING SELESAI! Akurasi Model: {eval_results['eval_accuracy'] * 100:.3f}%")
print("="*40 + "\n")

# ==========================================
# 7. SIMPAN BEST MODEL KE saved_models/codebert
# ==========================================
import os
import shutil

save_dir = "./saved_models/codebert"
os.makedirs(save_dir, exist_ok=True)

print("9. Menyimpan best model dan tokenizer...")
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"✅ Best model & tokenizer disimpan di: {save_dir}")

# Hapus folder checkpoint sementara untuk hemat disk
temp_dir = "./codebert-temp-checkpoints"
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
    print(f"🗑️  Folder checkpoint sementara ({temp_dir}) dihapus.")

# ==========================================
# 8. TES PREDIKSI DENGAN KODE CONTOH
# ==========================================
def predict_code(code_string):
    """Fungsi untuk menebak apakah sebuah kode buatan AI atau Manusia"""
    inputs = tokenizer(code_string, return_tensors="pt", truncation=True, max_length=512, padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    
    return "🤖 AI-Generated Code (Label 1)" if prediction == 1 else "👨‍💻 Human-Written Code (Label 0)"

print("\nMencoba melakukan prediksi pada kode contoh...")

contoh_kode_ai = """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
"""
print(f"Hasil prediksi contoh kode:\n{predict_code(contoh_kode_ai)}")