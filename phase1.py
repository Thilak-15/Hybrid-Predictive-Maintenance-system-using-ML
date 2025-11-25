import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sklearn stuff for clustering and data wrangling
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ----------- Load Data ------------------

# NOTE: The file uses space as delimiter but includes weird trailing separators
# so we're going to read it a bit crudely.
df_raw = pd.read_csv(r'CMaps\train_FD001.txt', sep=' ', header=None)

# Just manually dropping the empty columns that show up due to space parsing quirk
df_raw.dropna(axis=1, how='all', inplace=True)

# ------ Sensor Columns (manual slicing) --------
# We assume sensor readings start from 3rd column onward, for now.
# Could verify later if needed...
sensor_columns = df_raw.iloc[:, 2:27].values

# ------ NaN Check --------
# Check for missing values just to be sure
missing_counts = np.isnan(sensor_columns).sum(axis=0)
print("Missing values count for each sensor column:")
print(missing_counts)

# Identify columns that are entirely NaN (rare but still... worth checking)
fully_missing_idxs = np.where(missing_counts == sensor_columns.shape[0])[0]
if len(fully_missing_idxs) > 0:
    print(f"Warning: Entirely NaN columns found -> {fully_missing_idxs}")
else:
    print("No fully empty columns (phew)")

# Drop fully empty columns before we impute
# Could probably just let imputer handle it, but better safe than sorry
if len(fully_missing_idxs) > 0:
    cleaned_sensor_data = np.delete(sensor_columns, fully_missing_idxs, axis=1)
else:
    cleaned_sensor_data = sensor_columns.copy()  # Just in case

# ------- Impute missing values (mean imputation) --------
# We're just going with mean here — simple and works okay for now
imputer = SimpleImputer(strategy='mean')
filled_sensor_data = imputer.fit_transform(cleaned_sensor_data)

# --------- Standardize the features (z-score basically) ---------
# This is necessary for PCA and KMeans to behave properly
scaler = StandardScaler()
normalized_sensor_data = scaler.fit_transform(filled_sensor_data)

# ------- Clustering --------
# We'll use KMeans with a fixed number of clusters
# Not the most robust method but it's a start
num_clusters = 5
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans_model.fit_predict(normalized_sensor_data)

# Attach cluster info back to original dataframe
# Not ideal to mix raw and processed data in same df, but okay for now
df_raw['Cluster'] = cluster_labels

# -------- PCA for Visualization --------
# Reduce dimensions for plotting
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(normalized_sensor_data)

# -------- Plotting --------
# Plotting clusters in PCA-reduced 2D space
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
            c=cluster_labels, cmap='viridis', alpha=0.6)
plt.title("Clusters from PCA'd Sensor Data")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label='Cluster ID')
plt.tight_layout()
plt.show()