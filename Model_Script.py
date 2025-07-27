import pandas as pd
import numpy as np
import os
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, classification_report
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Step 1: Load from existing Power BI dataset input
df = dataset.copy()

# Step 2: Clean - drop empty columns and constant-value columns
df.dropna(axis=1, how='all', inplace=True)
df = df.loc[:, df.nunique() > 1]
df.dropna(inplace=True)

# Step 3: Label encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 4: Automatically detect churn column
target_column = None
for col in df.columns:
    if 'churn' in col.lower():
        target_column = col
        break

if not target_column:
    raise Exception("❌ No column with 'churn' in its name found.")

# Step 5: Split into features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Train XGBoost model with grid search
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}
grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_grid,
    cv=3,
    scoring='accuracy'
)
grid.fit(X_train, y_train)
model = grid.best_estimator_

# Step 8: Predict on full data
df['Predicted_Churn'] = model.predict(X)
df['Churn_Probability'] = model.predict_proba(X)[:, 1]

# Step 9: Add evaluation metrics as new columns
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]

df['Accuracy'] = accuracy_score(y_test, y_pred_test)
df['Precision'] = precision_score(y_test, y_pred_test)
df['Recall'] = recall_score(y_test, y_pred_test)
df['ROC_AUC'] = roc_auc_score(y_test, y_proba_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
df['TN'] = tn
df['FP'] = fp
df['FN'] = fn
df['TP'] = tp

# Step 10: Return updated dataset to Power BI
dataset = df
