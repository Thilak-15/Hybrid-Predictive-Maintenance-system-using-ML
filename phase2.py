import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import joblib

# ----------- Load and Prep the Dataset -----------

# Note: make sure the path actually points to the data file...
data_path = r'CMaps\train_FD001.txt'
raw_df = pd.read_csv(data_path, sep=' ', header=None)

# The actual sensor data spans cols 2 to 26 (inclusive)
sensor_cols = raw_df.iloc[:, 2:27].values

# Let's check for any columns that are entirely NaNs (I've seen this before with weird files)
all_nan_indices = np.where(np.all(np.isnan(sensor_cols), axis=0))[0]

if len(all_nan_indices) > 0:
    # Drop completely NaN columns - they won't help much...
    print("Columns with only NaNs:", all_nan_indices)
    sensor_cleaned = np.delete(sensor_cols, all_nan_indices, axis=1)
else:
    print("No full-NaN columns found.")
    sensor_cleaned = sensor_cols  # no change, just aliasing

print("Cleaned sensor data shape:", sensor_cleaned.shape)

# ----------- Handle Missing Values -----------

# Mean imputation feels reasonable here, but maybe we revisit if needed
mean_imputer = SimpleImputer(strategy='mean')
sensor_filled = mean_imputer.fit_transform(sensor_cleaned)

print("After imputation shape:", sensor_filled.shape)

# ----------- Standardize Features -----------

scaler = StandardScaler()
sensor_scaled = scaler.fit_transform(sensor_filled)

# ----------- Clustering to Assign Stages -----------

# Let's go with 5 clusters — seems like a decent number to capture degradation stages
cluster_model = KMeans(n_clusters=5, random_state=42)
cluster_labels = cluster_model.fit_predict(sensor_scaled)

# Add these pseudo-labels back into the original dataframe (weird but necessary)
raw_df['Stage_Label'] = cluster_labels

# ----------- Prepare Data for Classifier -----------

X_data = sensor_scaled
y_labels = raw_df['Stage_Label'].values

# Stratified split so each stage shows up in train/test
X_tr, X_te, y_tr, y_te = train_test_split(
    X_data, y_labels, test_size=0.2, stratify=y_labels, random_state=42
)

# ----------- Train the Classifier -----------

# Going with logistic regression because... well, it's quick and does decently
# Using class_weight='balanced' to handle any cluster imbalance
log_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_model.fit(X_tr, y_tr)

# ----------- Evaluate Results -----------

y_guess = log_model.predict(X_te)

print("=== Classification Report ===")
# Naming the stages for readability — definitely helps when checking output
stage_names = [f'Stage {i}' for i in range(5)]
print(classification_report(y_te, y_guess, target_names=stage_names))

# ----------- Confusion Matrix Plot -----------

conf_mat = confusion_matrix(y_te, y_guess)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=stage_names,
            yticklabels=stage_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Stage Classifier')
plt.tight_layout()
plt.show()

# ----------- Plot Feature Importance -----------

# We'll just use absolute value of coefficients — not perfect, but gives a rough idea
# (Averaging across classes just for sanity)
coef_importance = np.mean(np.abs(log_model.coef_), axis=0)

# Naming sensors from 1 onward, even though it's zero-indexed
feature_names = [f'Sensor {i+1}' for i in range(X_data.shape[1])]
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': coef_importance})

# Sort by importance descending
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='mako')
plt.title('Feature Importance (Logistic Regression)')
plt.tight_layout()
plt.show()

# ----------- Save Artifacts for Later Use -----------

# Saving the model and test data for Phase 4 — should maybe wrap this in a function later
joblib.dump(log_model, 'classification_model.joblib')
joblib.dump(X_te, 'X_test.joblib')
joblib.dump(y_te, 'y_test_stage.joblib')
