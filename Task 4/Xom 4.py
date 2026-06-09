# ==========================================================
# TASK 4: LOAN DEFAULT RISK + DATASET GENERATION
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

# ==========================================================
# 0. CREATE SYNTHETIC DATASET (500 ROWS)
# ==========================================================

np.random.seed(42)
n = 500

age = np.random.randint(21, 65, n)
income = np.random.normal(60000, 20000, n).clip(10000, 200000)
loan_amount = np.random.normal(20000, 10000, n).clip(1000, 100000)
credit_score = np.random.randint(300, 850, n)
employment_years = np.random.randint(0, 40, n)
num_dependents = np.random.randint(0, 5, n)

debt_ratio = loan_amount / (income + 1)

# Synthetic risk logic (target creation)
risk_score = (
    (loan_amount / income) * 0.4 +
    (1 - (credit_score - 300) / 550) * 0.4 +
    (num_dependents / 5) * 0.1 +
    np.random.normal(0, 0.05, n)
)

risk_score = np.clip(risk_score, 0, 1)
target = (risk_score > 0.5).astype(int)

df = pd.DataFrame({
    "age": age,
    "income": income.astype(int),
    "loan_amount": loan_amount.astype(int),
    "credit_score": credit_score,
    "employment_years": employment_years,
    "num_dependents": num_dependents,
    "debt_ratio": debt_ratio,
    "risk_score": risk_score,
    "TARGET": target
})

# Save dataset
df.to_csv("loan_default_synthetic_500.csv", index=False)
print("Dataset created: loan_default_synthetic_500.csv")

# ==========================================================
# 1. LOAD DATA (NOW FROM GENERATED CSV)
# ==========================================================

df = pd.read_csv("loan_default_synthetic_500.csv")

y = df["TARGET"]
X = df.drop(columns=["TARGET", "risk_score"])  # remove leakage column

# ==========================================================
# 2. TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# 3. PREPROCESSING
# ==========================================================

preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_prep = preprocess.fit_transform(X_train)
X_test_prep = preprocess.transform(X_test)

# ==========================================================
# 4. LOGISTIC REGRESSION MODEL
# ==========================================================

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_prep, y_train)

log_probs = log_model.predict_proba(X_test_prep)[:, 1]
print("Logistic AUC:", roc_auc_score(y_test, log_probs))

# ==========================================================
# 5. CATBOOST MODEL
# ==========================================================

cat_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    verbose=0
)

cat_model.fit(X_train_prep, y_train)

cat_probs = cat_model.predict_proba(X_test_prep)[:, 1]
print("CatBoost AUC:", roc_auc_score(y_test, cat_probs))

# ==========================================================
# 6. BUSINESS COST FUNCTION
# ==========================================================

COST_FN = 1000
COST_FP = 100

def calculate_cost(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    return fp * COST_FP + fn * COST_FN

# ==========================================================
# 7. THRESHOLD OPTIMIZATION
# ==========================================================

thresholds = np.linspace(0.05, 0.95, 50)

costs = []

for t in thresholds:
    costs.append(calculate_cost(y_test.values, cat_probs, t))

best_threshold = thresholds[np.argmin(costs)]

print("\nBest Threshold:", best_threshold)
print("Minimum Cost:", min(costs))

# ==========================================================
# 8. FEATURE IMPORTANCE
# ==========================================================

importances = cat_model.get_feature_importance()
features = X.columns

feat_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10,5))
plt.bar(feat_df["feature"], feat_df["importance"])
plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# ==========================================================
# 9. FINAL SUMMARY
# ==========================================================

print("\n================ BUSINESS SUMMARY ================")
print("Optimal Threshold:", best_threshold)
print("Minimum Business Cost:", min(costs))
print("Dataset saved as: loan_default_synthetic_500.csv")
print("==================================================")