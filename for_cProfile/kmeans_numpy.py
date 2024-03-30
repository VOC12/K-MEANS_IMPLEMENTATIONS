import numpy as np
import random
import cProfile
import pstats

# Asegúrate de tener NumPy instalado
def euclidean_distance(p1, p2):
    #Calculate the Euclidean distance between two points
    return np.sqrt(np.sum((p1 - p2) ** 2))

def initialize_centroids(data, k):
    #Initialize centroids by randomly selecting k data points
    centroids_idx = random.sample(range(len(data)), k)
    centroids = data[centroids_idx]
    return centroids

def kmeans2(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        distances = np.array([[euclidean_distance(point, centroid) for centroid in centroids] for point in data])
        nearest_centroid_indices = np.argmin(distances, axis=1)
        for i, idx in enumerate(nearest_centroid_indices):
            clusters[idx].append(data[i])
        new_centroids = np.array([np.mean(cluster, axis=0) if cluster else data[random.randint(0, len(data) - 1)] for cluster in clusters])
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, clusters

if __name__ == "__main__":
    data = np.random.rand(100, 2) * 100
    k = 3
    profiler = cProfile.Profile()
    profiler.enable()
    centroids, clusters = kmeans2(data, k)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('time')
    stats.print_stats()

    print("Centroids:", centroids)
