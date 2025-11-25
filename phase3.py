import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib

# ===== Load and preprocess data =====
file_path = r'CMaps\train_FD001.txt'  # Adjust path if needed
raw_df = pd.read_csv(file_path, sep=' ', header=None)

# Pull out engine ID, cycle count, and the sensor readings
eng_ids = raw_df.iloc[:, 0].values
cycle_vals = raw_df.iloc[:, 1].values
sensor_readings = raw_df.iloc[:, 2:27].values

# Detect sensor columns that are totally NaN (yes, this happens...)
nan_cols = np.where(np.isnan(sensor_readings).all(axis=0))[0]
print(f"Completely missing sensor columns: {nan_cols}")

# Drop them for now — no point keeping dead weight
if len(nan_cols) > 0:
    cleaned_sensors = np.delete(sensor_readings, nan_cols, axis=1)
else:
    cleaned_sensors = sensor_readings.copy()

print(f"Sensor data shape after removing NaN columns: {cleaned_sensors.shape}")

# === Fill missing values ===
# Note: mean imputation for now — not ideal but fast
imputer = SimpleImputer(strategy='mean')
sensors_filled = imputer.fit_transform(cleaned_sensors)

print(f"Shape after filling missing values: {sensors_filled.shape}")

# Normalize the data — mostly just for clustering & ML models
scaler = StandardScaler()
sensors_scaled = scaler.fit_transform(sensors_filled)

# === Cluster into degradation stages ===
num_stages = 5
kmeans_model = KMeans(n_clusters=num_stages, random_state=42)
stage_labels = kmeans_model.fit_predict(sensors_scaled)
raw_df['Stage'] = stage_labels

# Prep DataFrame for easier manipulation later
df_main = pd.DataFrame({
    'EngineID': eng_ids,
    'Cycle': cycle_vals,
    'Stage': stage_labels
})

# === Estimate time until next degradation stage ===
# Not super efficient — revisit if needed
time_left = []
for eid in df_main['EngineID'].unique():
    temp = df_main[df_main['EngineID'] == eid].sort_values('Cycle')
    stage_series = temp['Stage'].values
    cycle_series = temp['Cycle'].values
    for idx in range(len(temp)):
        curr_stage = stage_series[idx]
        curr_cycle = cycle_series[idx]
        delta = np.nan  # assume nothing unless found
        next_stages = np.where(stage_series > curr_stage)[0]
        later_stages = next_stages[next_stages > idx]  # ignore history
        if len(later_stages) > 0:
            jump = later_stages[0]
            delta = cycle_series[jump] - curr_cycle
        time_left.append(delta)

df_main['TimeLeft'] = time_left

# Drop rows without a valid label (i.e., last known stage for each engine)
df_valid = df_main.dropna(subset=['TimeLeft']).reset_index(drop=True)

# Grab corresponding feature vectors
X = sensors_scaled[df_valid.index]
y = df_valid['TimeLeft'].values

# === Train-test split ===
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
# RandomForest for quick prototyping — might switch to XGBoost later
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_tr, y_tr)

# === Predictions & Metrics ===
y_hat = rf_model.predict(X_te)

# Evaluate (quick metrics)
print("Model Evaluation:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_te, y_hat)):.3f}")
print(f"  MAE : {mean_absolute_error(y_te, y_hat):.3f}")
print(f"  R2  : {r2_score(y_te, y_hat):.3f}")

# === Visuals ===
plt.figure(figsize=(8, 6))
plt.scatter(y_te, y_hat, alpha=0.45, c='blue')
plt.plot([min(y_te), max(y_te)], [min(y_te), max(y_te)], 'r--')
plt.xlabel('Actual Time to Next Stage')
plt.ylabel('Predicted Time to Next Stage')
plt.title('Actual vs Predicted TTF (by RandomForest)')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Feature Importance ===
importance_scores = rf_model.feature_importances_
sensor_labels = [f'Sensor {i+1}' for i in range(X.shape[1])]

imp_df = pd.DataFrame({'Feature': sensor_labels, 'Importance': importance_scores})
imp_df = imp_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df, x='Importance', y='Feature')
plt.title('Sensor Importance (Random Forest)')
plt.tight_layout()
plt.show()

# === Save model & test data ===
# Probably useful for inference or Phase 4
joblib.dump(rf_model, 'regression_model.joblib')
joblib.dump(X_te, 'X_test_reg.joblib')
joblib.dump(y_te, 'y_test_time.joblib')