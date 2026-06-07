# 🤖 AICodeTrace - Model Directory

Direktori ini berisi skrip pelatihan (*training scripts*) untuk mendeteksi apakah suatu kode sumber ditulis oleh **Manusia (Human-written)** atau **Kecerdasan Buatan (AI-generated)**. Deteksi pada model-model ini difokuskan **khusus untuk bahasa pemrograman Python**.

---

## 🔗 Tautan Akses Web & API
* **Backend API (Live Hugging Face Space)**: [https://thomatomb-aicodetrace-backend.hf.space](https://thomatomb-aicodetrace-backend.hf.space)
* **Frontend Web Application**: [https://code-detection-8sml.onrender.com](https://code-detection-8sml.onrender.com)

---

## 📊 Dataset & Spesifikasi Model

Dataset yang digunakan untuk melatih kedua model di bawah ini adalah [basakdemirok/AIGCodeSet](https://huggingface.co/datasets/basakdemirok/AIGCodeSet) dari Hugging Face Hub. Dataset ini berisi snippet kode Python yang diklasifikasikan dengan label:
* **`0`**: Ditulis oleh Manusia (Human-written)
* **`1`**: Dibuat oleh AI (AI-generated)

---

## 🔍 Dokumentasi Proses Deteksi Kode

Proyek ini menggunakan dua pendekatan model yang berbeda untuk menganalisis kode Python:

### 1. Model Deep Learning: CodeBERT (`codebert.py`)
CodeBERT adalah model bahasa pra-latih (*pre-trained language model*) yang dirancang khusus untuk memahami representasi kode pemrograman (*code representation*).

* **Model Dasar**: `microsoft/codebert-base`
* **Arsitektur**: `AutoModelForSequenceClassification` dengan 2 label output.
* **Proses Deteksi**:
  1. **Tokenisasi**: Kode Python mentah dipotong menjadi token menggunakan tokenizer CodeBERT. Kode dibatasi maksimal **512 token** (`max_length=512`), dengan pemotongan (*truncation*) untuk kode panjang dan penambahan token kosong (*padding*) untuk kode pendek.
  2. **Inference**: Token dimasukkan ke dalam model Transformer untuk mengekstraksi makna semantik dari struktur dan sintaksis kode.
  3. **Output**: Model menghasilkan nilai *logits* untuk kelas 0 dan 1. Logits diproses menggunakan fungsi **Softmax** untuk menghasilkan probabilitas/tingkat keyakinan (*confidence score*), dan indeks probabilitas tertinggi diambil sebagai keputusan prediksi (*verdict*).

---

### 2. Model ML Klasik: TF-IDF + Logistic Regression (`ml2.py`)
Pendekatan ini menggabungkan representasi tekstual (kata-kata kunci) dengan fitur struktural (gaya penulisan) dari kode Python.

* **Proses Deteksi**:
  Proses deteksi pada model ini melalui beberapa tahap transformasi sebelum diklasifikasikan oleh Logistic Regression:
  
  ```
  [Kode Python] ──┬──► TF-IDF Vectorizer (max_features=3000) ────┬──► hstack() ──► Logistic Regression ──► Hasil Prediksi
                  │                                              ▲
                  └──► Ekstraksi 5 Fitur Meta ──► StandardScaler ─┘
  ```

  1. **Ekstraksi Fitur Tekstual (TF-IDF)**:
     Kode ditransformasikan menjadi representasi numerik menggunakan `TfidfVectorizer` dengan membatasi maksimal 3.000 fitur kata kunci terpenting. Stop words tidak dibuang agar pola sintaksis unik (seperti `for`, `if`, `while`) tetap terekam.
  
  2. **Ekstraksi 5 Fitur Struktural (Meta Features)**:
     Model mengekstrak metrik gaya penulisan Python yang sering kali berbeda antara manusia dan AI:
     * **`lines`**: Total seluruh baris di dalam file kode.
     * **`code_lines`**: Baris yang berisi kode (bukan baris kosong atau komentar).
     * **`comments`**: Baris yang diawali dengan karakter komentar Python (`#`).
     * **`functions`**: Jumlah definisi fungsi Python yang dideteksi menggunakan pola kecocokan regex `def <nama_fungsi>`.
     * **`blank_lines`**: Baris kosong di dalam kode.
  
  3. **Penyelarasan Skala (Scaling)**:
     Nilai dari 5 fitur struktural di atas diselaraskan skalanya menggunakan `StandardScaler` agar memiliki bobot yang seimbang dengan bobot nilai TF-IDF.
  
  4. **Penggabungan Fitur & Klasifikasi**:
     Matriks TF-IDF dan matriks fitur struktural yang telah discaling digabungkan secara horizontal (`scipy.sparse.hstack`) menjadi satu representasi vektor utuh, lalu dimasukkan ke dalam model **Logistic Regression** untuk menghasilkan prediksi kelas beserta tingkat keyakinannya (`predict_proba`).

---

## 🚀 Cara Menjalankan Pelatihan

### 1. Prasyarat (Requirements)
Pastikan dependensi Python berikut sudah terinstal di lingkungan kerja Anda:
```bash
pip install torch transformers datasets evaluate scikit-learn joblib scipy pandas numpy
```

### 2. Memeriksa Ketersediaan GPU (CUDA)
Sebelum melatih model Deep Learning (CodeBERT), disarankan untuk memverifikasi apakah GPU Anda terdeteksi oleh PyTorch:
```bash
python test.py
```

### 3. Melatih Model ML Klasik (TF-IDF + Logistic Regression)
Jalankan perintah berikut untuk melatih model Logistic Regression. Hasil model, vectorizer, dan scaler akan disimpan di direktori `saved_models/ml2/`:
```bash
python ml2.py
```

### 4. Melatih Model Deep Learning (CodeBERT)
Jalankan perintah berikut untuk melakukan *fine-tuning* model CodeBERT pada dataset. Hasil pelatihan terbaik akan disimpan di direktori `saved_models/codebert/`:
```bash
python codebert.py
```
> ⚠️ **Penting**: Kode `codebert.py` secara default mengaktifkan parameter `fp16=True` demi efisiensi VRAM. Jika Anda melatih model menggunakan CPU (tanpa GPU CUDA), silakan buka berkas `codebert.py` dan ubah parameter tersebut menjadi `fp16=False` pada `TrainingArguments` untuk menghindari crash.
