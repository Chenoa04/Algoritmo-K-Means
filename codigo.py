import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gap_statistic import OptimalK  # Requiere instalación: pip install gap-stat

# Configuración
np.random.seed(42)
plt.style.use('seaborn')

# 1. Generación de datos de ejemplo
datos, _ = make_blobs(n_muestras=500, centros=4, desviacion_cluster=0.8, random_state=42)

# 2. Método del Codo (Elbow Method)
def metodo_codo(datos, max_k=10):
    suma_cuadrados_intra = []
    for k in range(1, max_k+1):
        kmedias = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmedias.fit(datos)
        suma_cuadrados_intra.append(kmedias.inertia_)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, max_k+1), suma_cuadrados_intra, 'bo-')
    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('Suma de Cuadrados Intra-Cluster (WCSS)')
    plt.title('Método del Codo')
    plt.grid(True)
    
    # Cálculo de la segunda derivada para encontrar el codo
    diferencias = np.diff(suma_cuadrados_intra, 2)
    k_optimo_codo = np.argmax(diferencias) + 2  # +2 porque la segunda derivada tiene n-2 elementos
    
    return k_optimo_codo

# 3. Método de Silhouette
def metodo_silueta(datos, max_k=10):
    puntajes_silueta = []
    for k in range(2, max_k+1):  # Silhouette no funciona con k=1
        kmedias = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        etiquetas = kmedias.fit_predict(datos)
        puntajes_silueta.append(silhouette_score(datos, etiquetas))
    
    plt.subplot(1, 3, 2)
    plt.plot(range(2, max_k+1), puntajes_silueta, 'go-')
    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('Puntaje Silueta')
    plt.title('Método Silueta')
    plt.grid(True)
    
    k_optimo_silueta = np.argmax(puntajes_silueta) + 2  # +2 porque empezamos en k=2
    
    return k_optimo_silueta

# 4. Método Gap Statistic
def metodo_gap(datos, max_k=10):
    optimoK = OptimalK(parallel_backend='rust')
    n_clusters = optimoK(datos, cluster_array=np.arange(1, max_k+1))
    
    plt.subplot(1, 3, 3)
    optimoK.gap_df.plot()
    plt.title('Método Gap Statistic')
    plt.grid(True)
    
    return n_clusters

# Ejecución de los métodos
max_k = 10
k_optimo_codo = metodo_codo(datos, max_k)
k_optimo_silueta = metodo_silueta(datos, max_k)
k_optimo_gap = metodo_gap(datos, max_k)

plt.tight_layout()
plt.show()

# Resultados
print(f"\nResultados:")
print(f"Método del Codo sugiere k = {k_optimo_codo}")
print(f"Método Silueta sugiere k = {k_optimo_silueta}")
print(f"Método Gap Statistic sugiere k = {k_optimo_gap}")

# Visualización final con el k óptimo (usando el método de mayoría)
k_optimo = max(set([k_optimo_codo, k_optimo_silueta, k_optimo_gap]), 
               key=[k_optimo_codo, k_optimo_silueta, k_optimo_gap].count)

kmedias = KMeans(n_clusters=k_optimo, init='k-means++', n_init=10, random_state=42)
etiquetas = kmedias.fit_predict(datos)

plt.figure(figsize=(8, 6))
plt.scatter(datos[:, 0], datos[:, 1], c=etiquetas, cmap='viridis', s=50, alpha=0.7)
plt.scatter(kmedias.cluster_centers_[:, 0], kmedias.cluster_centers_[:, 1], 
            c='red', s=200, marker='X', label='Centroides')
plt.title(f'Clustering Final (k={k_optimo})')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True)
plt.show()
