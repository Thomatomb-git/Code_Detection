# 🤖 AICodeTrace - Model Directory

This directory contains the training scripts for detecting whether a piece of source code was written by a **Human (Human-written)** or by **Artificial Intelligence (AI-generated)**. Detection with these models is focused **specifically on the Python programming language**.

---

## 🔗 Web & API Access Links
* **Backend API (Live Hugging Face Space)**: [https://thomatomb-aicodetrace-backend.hf.space](https://thomatomb-aicodetrace-backend.hf.space)
* **Frontend Web Application**: [https://code-detection-8sml.onrender.com](https://code-detection-8sml.onrender.com)

---

## 📊 Dataset & Model Specifications

The dataset used to train both models below is [basakdemirok/AIGCodeSet](https://huggingface.co/datasets/basakdemirok/AIGCodeSet) from Hugging Face Hub. This dataset contains Python code snippets classified with the following labels:
* **`0`**: Human-written
* **`1`**: AI-generated

---

## 🔍 Code Detection Process Documentation

This project uses two different model approaches to analyze Python code:

### 1. Deep Learning Model: CodeBERT (`codebert.py`)
CodeBERT is a pre-trained language model designed specifically to understand programming code representations.

* **Base Model**: `microsoft/codebert-base`
* **Architecture**: `AutoModelForSequenceClassification` with 2 output labels.
* **Detection Process**:
  1. **Tokenization**: Raw Python code is split into tokens using the CodeBERT tokenizer. Code is limited to a maximum of **512 tokens** (`max_length=512`), with truncation for long code and padding for short code.
  2. **Inference**: Tokens are fed into the Transformer model to extract semantic meaning from the code's structure and syntax.
  3. **Output**: The model produces logit values for class 0 and class 1. Logits are processed using a **Softmax** function to generate a probability/confidence score, and the index with the highest probability is taken as the prediction verdict.

---

### 2. Classic ML Model: TF-IDF + Logistic Regression (`ml2.py`)
This approach combines textual representation (keywords) with structural features (writing style) of Python code.

* **Detection Process**:
  The detection process in this model goes through several transformation stages before being classified by Logistic Regression:
  
  ```
  [Python Code] ──┬──► TF-IDF Vectorizer (max_features=3000) ────┬──► hstack() ──► Logistic Regression ──► Prediction Result
                  │                                               ▲
                  └──► Extract 5 Meta Features ──► StandardScaler ─┘
  ```

  1. **Textual Feature Extraction (TF-IDF)**:
     Code is transformed into a numerical representation using `TfidfVectorizer`, limiting it to a maximum of 3,000 most important keyword features. Stop words are not removed so that unique syntactic patterns (such as `for`, `if`, `while`) are still captured.
  
  2. **5 Structural Feature Extraction (Meta Features)**:
     The model extracts Python writing style metrics that often differ between human and AI:
     * **`lines`**: Total number of lines in the code file.
     * **`code_lines`**: Lines containing code (excluding blank lines or comments).
     * **`comments`**: Lines starting with the Python comment character (`#`).
     * **`functions`**: Number of Python function definitions detected using the regex pattern `def <function_name>`.
     * **`blank_lines`**: Blank lines in the code.
  
  3. **Feature Scaling**:
     The values of the 5 structural features above are scaled using `StandardScaler` to ensure balanced weighting with the TF-IDF values.
  
  4. **Feature Concatenation & Classification**:
     The TF-IDF matrix and the scaled structural feature matrix are concatenated horizontally (`scipy.sparse.hstack`) into a single complete vector representation, then fed into a **Logistic Regression** model to produce a class prediction along with its confidence level (`predict_proba`).

---

## 📈 Model Evaluation Results

Both models were evaluated on a test set of **1,517 samples** (951 Human, 566 AI). Below are the confusion matrices and classification reports for each model.

### 1. CodeBERT

**Confusion Matrix:**

| | Predicted **Human** | Predicted **AI** |
|---|:---:|:---:|
| **Actual Human** | 772 (TN) | 179 (FP) |
| **Actual AI** | 199 (FN) | 367 (TP) |

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| Human (0) | 0.80 | 0.81 | 0.80 | 951 |
| AI (1) | 0.67 | 0.65 | 0.66 | 566 |
| **Accuracy** | | | **0.75** | **1517** |
| Macro Avg | 0.73 | 0.73 | 0.73 | 1517 |
| Weighted Avg | 0.75 | 0.75 | 0.75 | 1517 |

---

### 2. Logistic Regression + TF-IDF

**Confusion Matrix:**

| | Predicted **Human** | Predicted **AI** |
|---|:---:|:---:|
| **Actual Human** | 881 (TN) | 70 (FP) |
| **Actual AI** | 395 (FN) | 171 (TP) |

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| Human (0) | 0.69 | 0.93 | 0.79 | 951 |
| AI (1) | 0.71 | 0.30 | 0.42 | 566 |
| **Accuracy** | | | **0.69** | **1517** |
| Macro Avg | 0.70 | 0.61 | 0.61 | 1517 |
| Weighted Avg | 0.70 | 0.69 | 0.65 | 1517 |

---

### Model Comparison Summary

| Metric | CodeBERT | Logistic Regression + TF-IDF |
|---|:---:|:---:|
| **Overall Accuracy** | **75%** | 69% |
| **AI Recall** | **65%** | 30% |
| **AI F1-Score** | **0.66** | 0.42 |
| **Human Recall** | 81% | **93%** |
| **Macro F1** | **0.73** | 0.61 |

**Key Findings:**
- **CodeBERT** delivers stronger overall performance and is more balanced in recognizing both classes (Human & AI), making it the recommended model for general use.
- **Logistic Regression + TF-IDF** is heavily biased toward the Human class — it achieves very high Human recall (93%) but very low AI recall (30%), meaning it frequently misclassifies AI-generated code as human-written (395 false negatives out of 566 AI samples).

---

## 🚀 How to Run Training

### 1. Prerequisites (Requirements)
Make sure the following Python dependencies are installed in your environment:
```bash
pip install torch transformers datasets evaluate scikit-learn joblib scipy pandas numpy
```

### 2. Check GPU Availability (CUDA)
Before training the Deep Learning model (CodeBERT), it is recommended to verify whether your GPU is detected by PyTorch:
```bash
python test.py
```

### 3. Train the Classic ML Model (TF-IDF + Logistic Regression)
Run the following command to train the Logistic Regression model. The resulting model, vectorizer, and scaler will be saved in the `saved_models/ml2/` directory:
```bash
python ml2.py
```

### 4. Train the Deep Learning Model (CodeBERT)
Run the following command to fine-tune the CodeBERT model on the dataset. The best training result will be saved in the `saved_models/codebert/` directory:
```bash
python codebert.py
```
> ⚠️ **Important**: The `codebert.py` script enables `fp16=True` by default for VRAM efficiency. If you are training the model on CPU (without a CUDA GPU), please open the `codebert.py` file and change this parameter to `fp16=False` in `TrainingArguments` to avoid crashes.
