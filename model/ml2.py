import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack # PENTING: Untuk menggabungkan sparse matrix
from datasets import load_dataset

datasets = load_dataset("basakdemirok/AIGCodeSet")
df = datasets['train'].to_pandas()

# Asumsi: dataframe kamu sudah tersimpan di variabel 'df'
# df = pd.read_csv(...) 

print("1. Memulai proses pembersihan data...")
# Kita buang 'ada_embedding' karena di file ini kita murni pakai TF-IDF dari teks 'code'
cols_to_drop = ['problem_id', 'submission_id', 'status_in_folder', 'LLM', 'ada_embedding']
df_clean = df.drop(columns=cols_to_drop, errors='ignore') 

# Pastikan tidak ada data kosong di kolom 'code' (mencegah error di TF-IDF)
df_clean['code'] = df_clean['code'].fillna('')

print("2. Proses Vectorization menggunakan TF-IDF...")
# max_features membatasi jumlah token/kata unik yang diambil. 
# Kita set misal 3000 agar prosesnya lebih ringan, tapi tetap mengambil pola kode terbanyak.
# stop_words=None karena kita TIDAK MAU membuang kata seperti "for", "if", "while" (itu penting di kode!)
vectorizer = TfidfVectorizer(max_features=3000, stop_words=None)
X_tfidf = vectorizer.fit_transform(df_clean['code'])

print("3. Memproses dan melakukan scaling pada fitur struktural (meta-features)...")
meta_cols = ['lines', 'code_lines', 'comments', 'functions', 'blank_lines']
X_meta = df_clean[meta_cols].values

# Scaling wajib dilakukan agar nilainya seimbang dengan bobot TF-IDF
scaler = StandardScaler()
X_meta_scaled = scaler.fit_transform(X_meta)

print("4. Menggabungkan matriks TF-IDF dengan fitur struktural...")
# Kita gunakan hstack dari scipy.sparse, BUKAN dari numpy!
# Ini menggabungkan sparse matrix (TF-IDF) dengan matriks biasa (fitur angka) dengan sangat efisien.
X_combined = hstack([X_tfidf, X_meta_scaled])
y = df_clean['label'].values

print("5. Membagi data menjadi Train (80%) dan Test (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y 
)

print("6. Melatih model Logistic Regression...")
model_tfidf = LogisticRegression(max_iter=2000, random_state=42)
model_tfidf.fit(X_train, y_train)

print("\n=== HASIL EVALUASI MODEL (TF-IDF) ===")
y_pred = model_tfidf.predict(X_test)

print(f"Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Human (0)", "AI (1)"]))

# ==========================================
# SIMPAN MODEL, VECTORIZER, DAN SCALER
# ==========================================
import joblib
import os

save_dir = "./saved_models/ml2"
os.makedirs(save_dir, exist_ok=True)

joblib.dump(model_tfidf, os.path.join(save_dir, "model.joblib"))
joblib.dump(vectorizer, os.path.join(save_dir, "vectorizer.joblib"))
joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))

print(f"\n✅ Model, Vectorizer, dan Scaler disimpan di: {save_dir}")
print(f"   - model.joblib (Logistic Regression)")
print(f"   - vectorizer.joblib (TfidfVectorizer)")
print(f"   - scaler.joblib (StandardScaler)")