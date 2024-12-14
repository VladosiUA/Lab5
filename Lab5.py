import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Завантаження даних
df = pd.read_csv('Mall_Customers.csv')

# Попередній огляд даних
print(df.head())
print(df.info())

# Перевірка пропущених значень
print(df.isnull().sum())

# Описова статистика
print(df.describe())

# Побудова гістограм для кожної змінної
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Стандартизація даних (крім категорійних змінних)
scaler = StandardScaler()
numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_features]), columns=numerical_features)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Метод ліктя
inertia = []
silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))

# Графік інерції
plt.figure(figsize=(10, 5))
plt.plot(K, inertia, marker='o')
plt.title('Метод ліктя')
plt.xlabel('Кількість кластерів')
plt.ylabel('Інерція')
plt.show()

# Графік коефіцієнта силуету
plt.figure(figsize=(10, 5))
plt.plot(K, silhouette_scores, marker='o', color='orange')
plt.title('Коефіцієнт силуету')
plt.xlabel('Кількість кластерів')
plt.ylabel('Silhouette Score')
plt.show()
# Обираємо оптимальну кількість кластерів (наприклад, 4)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Візуалізація кластерів
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster', palette='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], 
            s=300, c='red', marker='X', label='Centroids')
plt.title('Результати кластеризації')
plt.legend()
plt.show()

# Середні значення показників для кожного кластера
cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)

# Аналіз характеристик кластерів
for cluster, row in cluster_summary.iterrows():
    print(f"Кластер {cluster}:")
    print(row)
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)

# Візуалізація кластерів DBSCAN
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='DBSCAN_Cluster', palette='deep', s=100)
plt.title('Результати DBSCAN')
plt.show()
input("\nНатисніть Enter, щоб завершити програму...")