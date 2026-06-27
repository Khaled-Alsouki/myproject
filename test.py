# Import basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Load data
housing = fetch_california_housing(as_frame=True)
df = housing.frame  # Convert to DataFrame
print(" Data loaded successfully!")
print(f"Shape: {df.shape}")
print(df.head())



# 1. General information
print("="*50)
print("1. Data Information:")
print(df.info())

print("\n" + "="*50)
print("2. Descriptive Statistics:")
print(df.describe())

print("\n" + "="*50)
print("3. Check for missing values:")
print(df.isnull().sum())

# 4. Target distribution (house price)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['MedHouseVal'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of House Prices')
plt.xlabel('House Price (in $100,000)')
plt.ylabel('Frequency')

# 5. Correlation matrix
plt.subplot(1, 2, 2)
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Between Features')
plt.tight_layout()
plt.show()



# Separate features (X) and target (y)
X = df.drop('MedHouseVal', axis=1)  # Features
y = df['MedHouseVal']  # Target

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f" Training data size: {X_train.shape}")
print(f" Testing data size: {X_test.shape}")



print("="*50)
print("🔹 PART 2: REGRESSION TASK")
print("="*50)

# 1. Baseline model - Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n Linear Regression Results:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_lr):.4f}")
print(f"R²:   {r2_score(y_test, y_pred_lr):.4f}")

# 2. Ridge and Lasso
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print("\n Ridge Regression Results:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_ridge):.4f}")
print(f"R²:   {r2_score(y_test, y_pred_ridge):.4f}")

print("\n Lasso Regression Results:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lasso)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_lasso):.4f}")
print(f"R²:   {r2_score(y_test, y_pred_lasso):.4f}")

# 3. Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
# Select only MedInc and HouseAge
cols_to_poly = ['MedInc', 'HouseAge']
X_train_poly = poly.fit_transform(X_train[cols_to_poly])
X_test_poly = poly.transform(X_test[cols_to_poly])

# Combine with remaining features
X_train_combined = np.hstack([X_train.drop(cols_to_poly, axis=1).values, X_train_poly])
X_test_combined = np.hstack([X_test.drop(cols_to_poly, axis=1).values, X_test_poly])

# Retrain
lr_poly = LinearRegression()
lr_poly.fit(X_train_combined, y_train)
y_pred_poly = lr_poly.predict(X_test_combined)

print("\n Linear Regression with Polynomial Features (MedInc, HouseAge):")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_poly)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_poly):.4f}")
print(f"R²:   {r2_score(y_test, y_pred_poly):.4f}")

# 4. Strongest features
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
})
print("\n Strongest features (Linear Regression):")
print(feature_importance.sort_values('Coefficient', key=abs, ascending=False))





print("\n" + "="*50)
print("🔹 PART 3: CLASSIFICATION TASK")
print("="*50)

# 1. Binary classification
median_price = y.median()
y_binary = (y > median_price).astype(int)  # 1 = Expensive, 0 = Affordable

# Split data for classification
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_bin, y_train_bin)
y_pred_log = log_reg.predict(X_test_bin)

print("\n Logistic Regression (Binary Classification):")
print(f"Accuracy:  {accuracy_score(y_test_bin, y_pred_log):.4f}")
print(f"Precision: {precision_score(y_test_bin, y_pred_log):.4f}")
print(f"Recall:    {recall_score(y_test_bin, y_pred_log):.4f}")
print(f"F1-Score:  {f1_score(y_test_bin, y_pred_log):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_bin, y_pred_log))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_bin, y_train_bin)
y_pred_dt = dt.predict(X_test_bin)

print("\n Decision Tree (Binary Classification):")
print(f"Accuracy:  {accuracy_score(y_test_bin, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test_bin, y_pred_dt):.4f}")
print(f"Recall:    {recall_score(y_test_bin, y_pred_dt):.4f}")
print(f"F1-Score:  {f1_score(y_test_bin, y_pred_dt):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_bin, y_pred_dt))

# 2. Multi-class classification (4 quartiles)
y_quartiles = pd.qcut(y, q=4, labels=[0, 1, 2, 3])  # 4 classes

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X, y_quartiles, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_multi, y_train_multi)
y_pred_rf = rf.predict(X_test_multi)

print("\n Random Forest (Multi-class Classification with 4 classes):")
print(f"Accuracy: {accuracy_score(y_test_multi, y_pred_rf):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_multi, y_pred_rf))




print("\n" + "="*50)
print("🔹 PART 4: CLUSTERING TASK")
print("="*50)

# 1. Normalize data and reduce dimensions
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f" Variance explained by first two components: {pca.explained_variance_ratio_.sum():.2%}")

# 2. Apply KMeans with k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)

# 3. Evaluation
print("\n KMeans Clustering (k=4):")
print(f"Inertia (SSE): {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels_kmeans):.4f}")

# Compare k=3,4,5
for k in [3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    print(f"\nk={k}:")
    print(f"  Inertia: {km.inertia_:.2f}")
    print(f"  Silhouette: {silhouette_score(X_scaled, labels):.4f}")

# 4. Visualization in PCA space
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.title('KMeans Clusters in PCA Space (k=4)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

# 5. Visualization on California map
plt.subplot(1, 3, 2)
plt.scatter(df['Longitude'], df['Latitude'], c=labels_kmeans, 
            cmap='viridis', alpha=0.6, s=10)
plt.title('Clusters on California Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# 6. Analyze clusters by location
plt.subplot(1, 3, 3)
for i in range(4):
    cluster_data = df[labels_kmeans == i]
    plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], 
                label=f'Cluster {i}', alpha=0.6, s=10)
plt.title('Clusters Colored by Group')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.tight_layout()
plt.show()




print("\n" + "="*50)
print("🔹 PART 5: COMPARATIVE ANALYSIS & HIERARCHICAL CLUSTERING")
print("="*50)

# 1. Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=4)
labels_hier = hierarchical.fit_predict(X_scaled)

print("\n Hierarchical Clustering (k=4):")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels_hier):.4f}")

# 2. Draw Dendrogram (small sample)
sample_size = 100  # Take a sample for speed
linkage_matrix = linkage(X_scaled[:sample_size], method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=20)
plt.title('Hierarchical Clustering Dendrogram (Sample)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# 3. Hyperparameter optimization using GridSearchCV
print("\n🔍 Hyperparameter Tuning with GridSearchCV:")
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0]
}
ridge_grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
ridge_grid.fit(X_train, y_train)
print(f"Best Ridge alpha: {ridge_grid.best_params_['alpha']}")
print(f"Best Ridge R²: {ridge_grid.best_score_:.4f}")

# 4. Key takeaways
print("\n" + "="*50)
print(" 3 KEY TAKEAWAYS:")
print("="*50)
print("1.  Regression: MedInc (median income) is the strongest predictor of house price")
print("2.  Classification: Supervised models give accurate results but require labeled data")
print("3.  Clustering: Clusters correspond strongly to geographic regions in California")