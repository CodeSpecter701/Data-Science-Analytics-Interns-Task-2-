import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

current_folder = os.path.dirname(os.path.abspath(__file__))

csv_file = os.path.join(
    current_folder,
    "Mall_Customers_500Rows.csv"
)

df = pd.read_csv(csv_file)

print("Dataset Loaded Successfully!\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)"
)

plt.title("Income vs Spending Score")
plt.show()

# =====================================
# FEATURE SELECTION
# =====================================

X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

# =====================================
# SCALE DATA
# =====================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =====================================
# K-MEANS CLUSTERING
# =====================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# =====================================
# CLUSTER VISUALIZATION
# =====================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1"
)

plt.title("Customer Segments")
plt.show()

# =====================================
# PCA VISUALIZATION
# =====================================

pca = PCA(n_components=2)

pca_data = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))

plt.scatter(
    pca_data[:,0],
    pca_data[:,1],
    c=df["Cluster"],
    cmap="viridis"
)

plt.title("PCA Cluster Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()

# =====================================
# CLUSTER SUMMARY
# =====================================

summary = df.groupby("Cluster")[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
].mean()

print("\nCluster Summary")
print(summary)

# =====================================
# SAVE OUTPUT
# =====================================

output_file = os.path.join(
    current_folder,
    "Customer_Segments.csv"
)

df.to_csv(output_file, index=False)

print("\nCustomer segmentation completed successfully!")
print(f"Output saved to: {output_file}")

# =====================================
# MARKETING STRATEGIES
# =====================================

print("\nMarketing Strategies")

print("""
Cluster 0:
High spending customers
Strategy: VIP offers and premium memberships

Cluster 1:
Moderate spending customers
Strategy: Loyalty rewards and personalized promotions

Cluster 2:
Low spending customers
Strategy: Discount campaigns and coupons

Cluster 3:
High income but low spending
Strategy: Targeted advertising and product recommendations

Cluster 4:
Balanced customers
Strategy: Seasonal offers and cross-selling
""")