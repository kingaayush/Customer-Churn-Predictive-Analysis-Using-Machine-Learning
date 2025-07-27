import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")

# Load the dataset (adjust this path as needed)
df = pd.read_csv("CustomerChurn.csv")

# --- CLEANING AND PREPARATION ---
# Standardize column names
df.columns = df.columns.str.strip().str.title().str.replace(" +", " ", regex=True)

# Drop rows with missing target
if 'Churn' not in df.columns:
    raise ValueError("❌ 'Churn' column not found in dataset.")
df.dropna(subset=['Churn'], inplace=True)

# Fill missing values
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna("Unknown")

for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

# Define preferred features (you can modify this list)
preferred_features = [
    'Customer Id', 'Loyaltyid', 'Senior Citizen', 'Partner', 'Dependents',
    'Tenure', 'Phone Service', 'Multiple Lines', 'Internet Service',
    'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
    'Streaming Tv', 'Streaming Movies', 'Contract', 'Paperless Billing',
    'Payment Method', 'Monthly Charges', 'Total Charges'
]

# Filter the actual features available in dataset
available_features = [col for col in preferred_features if col in df.columns]

# Separate features and target
X = df[available_features].copy()
y = df['Churn']

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
label_encoders['Churn'] = target_encoder

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
print("\n✅ Target class distribution:")
print(pd.Series(y).value_counts(normalize=True).rename("proportion"))

print("\n📊 Model Evaluation:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model, encoders, and expected features
joblib.dump(model, "churn_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(available_features, "expected_features.pkl")
print("\n✅ Model, encoders, and expected feature list saved.")
