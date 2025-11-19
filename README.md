# ⭐ **Task 4 — Classification with Logistic Regression**

This repository contains **Task 4** of my AIML Internship project.
The objective of this task is to build a **binary classification model** using **Logistic Regression** on the Breast Cancer Wisconsin Dataset.

The model predicts whether a tumor is **Malignant (M)** or **Benign (B)** based on multiple diagnostic features.

---

## 📁 **Repository Structure**

```
├── data.csv                   # Raw dataset (Breast Cancer Wisconsin)
├── processed_dataset.csv      # Cleaned & preprocessed dataset
├── logistic.py                # Complete one-click runnable training script
├── README.md                  # Documentation
└── outputs/
    ├── model_lr.joblib              # Trained Logistic Regression model
    ├── scaler.joblib                # Fitted StandardScaler
    ├── test_summary.json            # Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    ├── classification_report.txt    # Detailed classification report
    ├── confusion_matrix.png         # Confusion Matrix heatmap
    ├── roc_curve.png                # ROC Curve plot
    ├── precision_recall_curve.png   # Precision–Recall Curve plot
```

---

## 🎯 **Objective**

Build and evaluate a **binary classifier** using Logistic Regression.
The model must:

* Preprocess the dataset
* Standardize numeric features
* Train/test split
* Fit Logistic Regression
* Tune the classification threshold
* Generate evaluation metrics and plots

---

## 🧹 **Data Preprocessing Steps**

1. Dropped irrelevant / empty columns:

   * `id`
   * `Unnamed: 32` (fully empty)

2. Converted target labels:

   * **M → 0 (Malignant)**
   * **B → 1 (Benign)**

3. Imputed missing values with **median**.

4. Scaled numerical features using **StandardScaler**.

5. Split into:

   * **70% Training**
   * **15% Validation**
   * **15% Testing**

6. Tuned classification threshold using **best F1-score**.

---

## 🤖 **Model Used**

### **Logistic Regression**

* `solver="liblinear"`
* `max_iter=2000`
* `class_weight="balanced"` (handles class imbalance)

---

## 📊 **Evaluation Metrics**

Stored inside `outputs/test_summary.json`:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **ROC-AUC Score**
* **Best validation threshold**

Additional evaluation outputs:

### ✔ Confusion Matrix

`outputs/confusion_matrix.png`

### ✔ ROC Curve

`outputs/roc_curve.png`

### ✔ Precision–Recall Curve

`outputs/precision_recall_curve.png`

### ✔ Full classification report

`outputs/classification_report.txt`

---

## 🧪 **How to Run the Project**

### **1️⃣ Google Colab (Recommended)**

Upload:

* `data.csv`
* `logistic.py`

Run:

```python
!python logistic.py
```

All outputs will be created automatically in:

```
/content/task4_outputs/
```

---

### **2️⃣ Local System**

**Install dependencies:**

```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib
```

**Run the script:**

```bash
python logistic.py
```

All outputs will appear in the `outputs/` directory.

---

## 📝 **Dataset Used**

**Breast Cancer Wisconsin (Diagnostic) Dataset**
Source: UCI Machine Learning Repository / sklearn datasets.

---

## ✨ **Author**

**Thrishool M S**

AIML Internship — *Task 4: Logistic Regression Classification*


