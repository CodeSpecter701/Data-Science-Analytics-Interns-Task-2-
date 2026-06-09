# ============================================================
# BANK TERM DEPOSIT SUBSCRIPTION PREDICTION PROJECT
# DevelopersHub Corporation - Advanced Internship Task
# ============================================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_curve,
    roc_auc_score
)

import shap
import joblib

# ============================================================
# STEP 1 - LOAD DATASET
# ============================================================

# Download from:
# https://archive.ics.uci.edu/ml/datasets/bank+marketing

df = pd.read_csv("bank.csv", sep=';')

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== DATA INFO ==========\n")
print(df.info())

print("\n========== DATA SHAPE ==========\n")
print(df.shape)

# ============================================================
# STEP 2 - DATA CLEANING
# ============================================================

# Convert target variable to numerical

df['y'] = df['y'].map({
    'yes': 1,
    'no': 0
})

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# ============================================================
# STEP 3 - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# -------------------------
# TARGET VARIABLE DISTRIBUTION
# -------------------------

plt.figure(figsize=(6,4))

sns.countplot(x='y', data=df)

plt.title("Term Deposit Subscription")
plt.xlabel("Subscribed")
plt.ylabel("Count")

plt.show()

# -------------------------
# HISTOGRAMS
# -------------------------

df.hist(figsize=(14,10))

plt.tight_layout()
plt.show()

# -------------------------
# CORRELATION HEATMAP
# -------------------------

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# ============================================================
# STEP 4 - SPLIT FEATURES & TARGET
# ============================================================

X = df.drop('y', axis=1)
y = df['y']

# ============================================================
# STEP 5 - IDENTIFY COLUMN TYPES
# ============================================================

categorical_cols = X.select_dtypes(include=['object']).columns

numerical_cols = X.select_dtypes(exclude=['object']).columns

print("\n========== CATEGORICAL COLUMNS ==========\n")
print(categorical_cols)

print("\n========== NUMERICAL COLUMNS ==========\n")
print(numerical_cols)

# ============================================================
# STEP 6 - PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# ============================================================
# STEP 7 - TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# STEP 8 - LOGISTIC REGRESSION MODEL
# ============================================================

log_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

print("\n========== TRAINING LOGISTIC REGRESSION ==========\n")

log_model.fit(X_train, y_train)

# Predictions

y_pred_log = log_model.predict(X_test)

y_prob_log = log_model.predict_proba(X_test)[:, 1]

# ============================================================
# STEP 9 - EVALUATE LOGISTIC REGRESSION
# ============================================================

print("\n========== LOGISTIC REGRESSION RESULTS ==========\n")

print("F1 Score:")
print(f1_score(y_test, y_pred_log))

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred_log))

# -------------------------
# CONFUSION MATRIX
# -------------------------

cm = confusion_matrix(y_test, y_pred_log)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Logistic Regression Confusion Matrix")

plt.show()

# -------------------------
# ROC CURVE
# -------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob_log)

plt.figure(figsize=(7,5))

plt.plot(fpr, tpr, label="Logistic Regression")

plt.plot([0,1], [0,1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

print("ROC-AUC Score:")
print(roc_auc_score(y_test, y_prob_log))

# ============================================================
# STEP 10 - RANDOM FOREST MODEL
# ============================================================

rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

print("\n========== TRAINING RANDOM FOREST ==========\n")

rf_model.fit(X_train, y_train)

# Predictions

y_pred_rf = rf_model.predict(X_test)

y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# ============================================================
# STEP 11 - EVALUATE RANDOM FOREST
# ============================================================

print("\n========== RANDOM FOREST RESULTS ==========\n")

print("F1 Score:")
print(f1_score(y_test, y_pred_rf))

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred_rf))

# -------------------------
# CONFUSION MATRIX
# -------------------------

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Random Forest Confusion Matrix")

plt.show()

# -------------------------
# ROC CURVE
# -------------------------

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(7,5))

plt.plot(fpr_rf, tpr_rf, label="Random Forest")

plt.plot([0,1], [0,1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Random Forest ROC Curve")

plt.legend()

plt.show()

print("ROC-AUC Score:")
print(roc_auc_score(y_test, y_prob_rf))

# ============================================================
# STEP 12 - SHAP EXPLAINABLE AI (XAI)
# ============================================================

print("\n========== SHAP EXPLAINABILITY ==========\n")

# Transform dataset

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

# Train separate RF model for SHAP

rf_explainer_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_explainer_model.fit(
    X_train_processed,
    y_train
)

# Create SHAP explainer

explainer = shap.TreeExplainer(
    rf_explainer_model
)

shap_values = explainer.shap_values(
    X_test_processed
)

# Feature names

feature_names = preprocessor.get_feature_names_out()

# ============================================================
# STEP 13 - SHAP SUMMARY PLOT
# ============================================================

print("\nGenerating SHAP Summary Plot...\n")

shap.summary_plot(
    shap_values[1],
    X_test_processed,
    feature_names=feature_names
)

# ============================================================
# STEP 14 - EXPLAIN 5 PREDICTIONS
# ============================================================

print("\n========== EXPLAINING 5 PREDICTIONS ==========\n")

for i in range(5):

    print(f"\nExplaining Prediction #{i+1}\n")

    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][i],
        X_test_processed[i],
        feature_names=feature_names,
        matplotlib=True
    )

    plt.show()

# ============================================================
# STEP 15 - FEATURE IMPORTANCE
# ============================================================

importances = rf_explainer_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\n========== TOP 10 IMPORTANT FEATURES ==========\n")

print(importance_df.head(10))

# Plot top 10 features

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df.head(10)
)

plt.title("Top 10 Important Features")

plt.show()

# ============================================================
# STEP 16 - SAVE MODEL
# ============================================================

joblib.dump(
    rf_model,
    "random_forest_bank_model.pkl"
)

print("\n========== MODEL SAVED SUCCESSFULLY ==========\n")

# ============================================================
# STEP 17 - BUSINESS INSIGHTS
# ============================================================

print("\n========== BUSINESS INSIGHTS ==========\n")

print("""
1. Customers with longer call durations are more likely to subscribe.

2. Previous successful marketing campaigns increase conversion probability.

3. Certain job categories respond better to term deposit offers.

4. Contact frequency influences customer decisions.

5. Economic indicators and campaign timing impact subscription rates.
""")

# ============================================================
# END OF PROJECT
# ============================================================

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========\n")