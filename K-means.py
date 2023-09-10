# Importación de las bibliotecas necesarias
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Lectura de los datos del CSV
data = pd.read_csv("./touristic_places_california.csv")

# ---------------- PREPROCESAMIENTO ---------------
# checando si hay valores faltantes (no hay)
data.isnull().sum()
# checando si hay valores `na` en el dataframe (no hay)
data.isna().sum()
# checando si hay duplicados (no hay)
duplicates = data[data.duplicated()]


# Selección de las columnas relevantes para el clustering
X = data[['Latitude', 'Longitude']]


# Normalización de los datos utilizando StandardScaler.
# Esto es importante para que K-Means funcione correctamente,
# ya que el algoritmo es sensible a la escala.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División en train y test (80% train, 20% test)
train_data, test_data = train_test_split(X_scaled, test_size=0.2,
                                         random_state=None)

# División adicional en train y validation (75% train, 25% validation)
train_data, validation_data = train_test_split(train_data, test_size=0.25,
                                               random_state=None)

# El código KMeans inicializa el modelo con 7 clústeres
# y None en random_state para generalizar.
kmeans = KMeans(n_clusters=7, random_state=None)
kmeans.fit(train_data)

print()
print("\nLlamada al método fit del modelo de K-Means que se usa para entrenar")
print("el algoritmo en un conjunto de datos escalados.")
print("    ", kmeans.fit(train_data), "\n")

print("Medida de la dispersión de los puntos dentro de los clústeres.")
print("  ", kmeans.inertia_)
print("Si el valor es más pequeño mejor será el ajuste del modelo, ya que")
print("indica que los puntos están más cerca de sus respectivos centroides.\n")

print("Locación de los centroides")
print("  ", kmeans.cluster_centers_, "\n")

print("Número de iteraciones que le tomó al modelo converger a una solución")
print("durante el ajuste.")
print("  ", kmeans.n_iter_, "\n")

print("Clusters asignados a los datos previo al ajuste de hiperparámetros:")
print("  ", kmeans.labels_, "\n")

# ------------------- CODO -------------------
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": None,
}
# La lista tiene los valores de Sum of Squared Errors para cada k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(train_data)
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
best_k = None
best_score = -1

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(train_data)
    validation_score = silhouette_score(validation_data,
                                        kmeans.predict(validation_data))
    silhouette_coefficients.append(validation_score)
    if validation_score > best_score:
        best_score = validation_score
        best_k = k
        best_model = kmeans
# Evalua el modelo con métricas de evaluación
silhouette_avg_test = silhouette_score(test_data,
                                       best_model.predict(test_data))
print("------ MÉTRICAS DE EVALUACIÓN ------- \n")
print("Elegí las métricas de evaluación de Silhouette porque es un indicador\
\nque se utiliza para evaluar la calidad de un clustering o agrupamiento\
\nde datos. Evalúa que tan bien definidos y separados están los clústeres\
\nen un conjunto de datos.\n")
print(f'Silhouette Score : {silhouette_avg_test}')
print("Como es un valor cercano a 1 significa que los clusters están bien\
    \nseparados y que cada dato está cerca de su cluster correspondiente. \n")

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

# Entrena el modelo final
final_kmeans = KMeans(n_clusters=7, random_state=None)
# Entrenar el modelo en los conjuntos de entrenamiento y validación
combined_train_data = np.vstack((train_data, validation_data))

final_kmeans.fit(combined_train_data)

# Obtiene predicciones
test_predictions = final_kmeans.predict(X_scaled)

# Métrica de evaluación de Inercia para saber su score y visualizarlo
inertia = final_kmeans.inertia_
print("\nElegí las métricas de evaluación de Inertia porque es  una métrica \
    \nusada para evaluar la calidad del algoritmo K-means. Refleja la suma de \
    \nlas distancias cuadradas de cada punto dentro de un clúster con respecto\
    \na su centroide. Cuanto menor sea la inercia, más compactos y \
    \ncohesionados serán los clústeres.\n")
print("Score de Inercia:", inertia, "\n")

visualizer = SilhouetteVisualizer(final_kmeans, colors='yellowbrick')
visualizer.fit(combined_train_data)
visualizer.show()
plt.show()

print("Valor de los clusters asignadas por el modelo:\n", test_predictions,
      "\n")

# -------------------Visualización de los resultados-------------------
# Realización de una reducción de dimensionalidad para visualizar los
# resultados. Uso de PCA para reducir la dimensionalidad de los datos a dos
# componentes principales. Pata visualizar los resultados
# del clustering en un gráfico bidimensional.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Agregación de las coordenadas PCA al DataFrame
data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]
# usé los valores obtenidos por el método de reducción de dimensionalidad PCA
# c toma el valor del resultado de la asignación de clústeres
plt.figure(num="K-Means Clustering", figsize=(10, 6))
plt.scatter(data['PCA1'], data['PCA2'], c=test_predictions, cmap='viridis')
plt.title('K-Means Clustering')
plt.show()

print("Decidí usar 7 clusters y centroides porque a pesar de que el método del\
\ncodo dijera que era mejor usar 2, el Score de Silhoutte indica que 7\
\nes buena opción, también estuve probando con diferentes valores y al\
\nvisualizarlo vi que con 7 quedaba bien separado. Para el parámetro init\
\ndeje el default que es k-means++ porque selecciona el centroide inicial\
\nde manera que optimiza el llegar a una solución de agrupación de calidad.\
\nDeje n_init en su default que es 10 ya que es el número de veces\
\nque el algorimto de K-means será ejecutado con\
\ndiferentes semillas de centroides y es recomendable un número alto El\
\nmax_iter lo deje en default que es 300. El paráetron de tol lo deje en 1e-4\
\nporque controla cuando se considera que el algoritmo ha llegado a una\
\nsolición y se detiene y si es un número muy pequeño entonces será más\
\npreciso. En random_state puse None para que la inicialización de los\
\ncentroides sea aleatoria en cada ejecución. En el parámetro de algorithm\
\ndeje el default que es lloyd ya que es el algoritmo clásico de K-means\
\nestilo EM que nos explicaron en clase.")

print()
