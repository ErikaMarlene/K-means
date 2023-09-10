# Importación de las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# Lectura de los datos del CSV
data = pd.read_csv("./touristic_places_california.csv")


# Selección de las columnas relevantes para el clustering
X = data[['Latitude', 'Longitude']]

# Normaliza los datos utilizando StandardScaler.
# Esto es importante para que K-Means funcione correctamente,
# ya que el algoritmo es sensible a la escala de las características.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# El código `kmeans = KMeans(n_clusters=7, random_state=42)` inicializa
# un modelo de clustering K-Means con 7 clústeres y un estado aleatorio de 42.
kmeans = KMeans(n_clusters=7, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Realiza una reducción de dimensionalidad para visualizar los resultados
# Utiliza PCA para reducir la dimensionalidad de los datos a dos
# componentes principales. Esto permitirá visualizar los resultados
# del clustering en un gráfico bidimensional.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Agrega las coordenadas PCA al DataFrame
data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]

print()
print("\nLlamada al método fit del modelo de K-Means que se usa para entrenar")
print("el algoritmo en un conjunto de datos escalados.")
print("    ", kmeans.fit(X_scaled), "\n")

print("Medida de la dispersión de los puntos dentro de los clústeres.")
print("  ", kmeans.inertia_)
print("Si el valor es más pequeño mejor será el ajuste del modelo, ya que")
print("indica que los puntos están más cerca de sus respectivos centroides.\n")

print("Locación de los centroides")
print("  ", kmeans.cluster_centers_, "\n")

print("Número de iteraciones que le tomó al modelo converger a una solución")
print("durante el ajuste.")
print("  ", kmeans.n_iter_, "\n")

print("Clusters asignados a los datos.")
print("  ", kmeans.labels_[:30], "\n")

# Método del CODO para ver una k probable
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
# La lista tiene los valores de Sum of Squared Errors para cada k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
plt.figure(num="CODO")
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse, marker='o', linestyle='-')
plt.xticks(range(1, 11))
plt.xlabel("Número de Clusters")
plt.ylabel("Inercia")
plt.show()


#  -------------------SILHOUETTE-------------------
# Calcula los coeficientes de silueta para diferentes valores de k
# (número de clústeres).
silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_coefficients.append(score)
plt.figure(num="SILHOUETTE")
plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients, marker='o', linestyle='-')
plt.xticks(range(2, 11))
plt.xlabel("Número de Clusters")
plt.ylabel("Coeficiente de Silhouette")
plt.show()

# -------------------SilhouetteVisualizer-------------------
# Con esta gráfica pude ajustar de mejor manera el valor de k ya que
# visualmente pude identificar el puntaje promedio y como casi todos los
# grupos rebasaban este promedio. Esto me hizo notar que los grupos que
# genera están bien separados.
model = KMeans(n_clusters=7, random_state=0)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
visualizer.fit(X_scaled)
visualizer.show()
plt.show()

# -------------------Visualización de los resultados-------------------
# usé los valores obtenidos por el método de reducción de dimensionalidad PCA
# c toma el valor del resultado de la asignación de clústeres
plt.figure(num="K-Means Clustering", figsize=(10, 6))
plt.scatter(data['PCA1'], data['PCA2'], c=kmeans.labels_, cmap='viridis')
plt.title('K-Means Clustering')
plt.show()

# print("  ",  , "\n")
print()
