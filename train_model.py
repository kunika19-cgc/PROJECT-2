import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle
import numpy as np

# 1. Load and prepare data
print("Loading data...")
df = pd.read_csv('heart.csv')
df = df.drop_duplicates()

# 2. Check data quality
print("\nData Overview:")
print(f"Total samples: {len(df)}")
print("Class Distribution:\n", df['target'].value_counts())
print("\nFirst 5 rows:\n", df.head())

# 3. Split features and target
X = df.drop('target', axis=1)
y = df['target']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

# 5. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print("\nAfter SMOTE:")
print(f"Resampled training data shape: {X_train_resampled.shape}")
print("Resampled class distribution:", np.bincount(y_train_resampled))

# 7. Train model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced_subsample'
)
model.fit(X_train_resampled, y_train_resampled)

# 8. Evaluate model
print("\nModel Evaluation:")
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

# 9. Save artifacts
print("\nSaving artifacts...")
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# 10. Verification test
print("\nRunning verification test...")
test_sample = X_test.iloc[0:1].values  # Get first test sample
print("\nTest sample features:", test_sample)

# Scale the test sample
test_sample_scaled = scaler.transform(test_sample)
prediction = model.predict(test_sample_scaled)
probability = model.predict_proba(test_sample_scaled)

print(f"\nVerification Result:")
print(f"Predicted class: {prediction[0]}")
print(f"Probability: {np.max(probability)*100:.2f}%")
print(f"Actual class: {y_test.iloc[0]}")

print("\nProcess completed successfully!")