# ---------------------------
# TASK 4 — Logistic Regression
# ---------------------------
import os, json, zipfile, joblib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    classification_report
)

# ---------------------------
# Locate dataset
# ---------------------------
path = "/content/data.csv"
if not os.path.exists(path):
    path = "/mnt/data/data.csv"
df = pd.read_csv(path)
print("Loaded:", path)

# ---------------------------
# CLEAN PROBLEM COLUMNS
# ---------------------------

# Drop ID column if present
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# Drop fully empty column `Unnamed: 32`
empty_cols = [c for c in df.columns if df[c].isnull().sum() == df.shape[0]]
if empty_cols:
    print("Dropping empty columns:", empty_cols)
    df.drop(columns=empty_cols, inplace=True)

# ---------------------------
# Detect target
# ---------------------------
if "diagnosis" in df.columns:
    target_col = "diagnosis"
else:
    raise ValueError("No diagnosis column found.")

# Map M→0, B→1 
df[target_col] = df[target_col].map({"M":0, "B":1})

# ---------------------------
# Impute any leftover NaNs BEFORE splitting
# ---------------------------
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# ---------------------------
# Split X and y
# ---------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# ---------------------------
# Train/Val/Test split
# ---------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ---------------------------
# Scale numeric features
# ---------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ---------------------------
# Train Logistic Regression
# ---------------------------
model = LogisticRegression(max_iter=2000, solver="liblinear", class_weight='balanced')
model.fit(X_train, y_train)

# ---------------------------
# Tune threshold using validation set
# ---------------------------
probs_val = model.predict_proba(X_val)[:,1]
best_thr = 0.5
best_f1 = -1

for t in np.linspace(0.05, 0.95, 50):
    preds = (probs_val >= t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = t

print(f"Best threshold: {best_thr:.3f}")

# ---------------------------
# Test predictions using tuned threshold
# ---------------------------
probs_test = model.predict_proba(X_test)[:,1]
y_pred = (probs_test >= best_thr).astype(int)

# ---------------------------
# Evaluation
# ---------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# Plot Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ---------------------------
# Plot ROC Curve
# ---------------------------
fpr, tpr, _ = roc_curve(y_test, probs_test)
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.show()

# ---------------------------
# Plot Precision-Recall Curve
# ---------------------------
prec, rec, _ = precision_recall_curve(y_test, probs_test)
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

print("\nDone")
