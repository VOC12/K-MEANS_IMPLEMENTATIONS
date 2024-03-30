import numpy as np
import kmeans_cythonc

def main():
    #datos aleatorios como un array de NumPy
    data = np.random.rand(100, 2) * 100  # 100 puntos 2D
    
    #clusters
    k = 3
    
    # Ejecuta el algoritmo K-Means usando tu implementación de Cython
    centroids, clusters = kmeans_cythonc.kmeans_cython(data, k)
    
    #resultados
    print("Centroids:")
    for centroid in centroids:
        print(centroid)
    
    print("\nClusters:")
    for cluster in clusters:
        print(cluster)

if __name__ == "__main__":
    main()
