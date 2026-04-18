# train_models.py
# Run this once to generate: model.pkl, rf_model.pkl, lr_model.pkl, svm_model.pkl
# Command: python train_models.py

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# ── Load dataset ──────────────────────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# Save as diabetes.csv in the same folder
df = pd.read_csv("diabetes.csv")

print("Dataset shape:", df.shape)
print(df["Outcome"].value_counts())

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Replace biologically impossible zeros with column median
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 1. Random Forest (Primary model → model.pkl) ──────────────────────────────
rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                             min_samples_leaf=4, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\n── Random Forest ──")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

pickle.dump(rf, open("model.pkl",    "wb"))   # primary (kept for backward compat)
pickle.dump(rf, open("rf_model.pkl", "wb"))

# ── 2. Logistic Regression ────────────────────────────────────────────────────
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])
lr_pipe.fit(X_train, y_train)
y_pred_lr = lr_pipe.predict(X_test)
print("\n── Logistic Regression ──")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr))

pickle.dump(lr_pipe, open("lr_model.pkl", "wb"))

# ── 3. SVM ────────────────────────────────────────────────────────────────────
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
])
svm_pipe.fit(X_train, y_train)
y_pred_svm = svm_pipe.predict(X_test)
print("\n── SVM ──")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(classification_report(y_test, y_pred_svm))

pickle.dump(svm_pipe, open("svm_model.pkl", "wb"))

print("\n✅ All models saved: model.pkl, rf_model.pkl, lr_model.pkl, svm_model.pkl")

