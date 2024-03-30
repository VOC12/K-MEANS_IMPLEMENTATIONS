import random
import math
import cProfile
import pstats

def euclidean_distance(p1, p2):
    # Calculate the Euclidean distance between two points.
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

def initialize_centroids(data, k):
    """Initialize centroids by randomly selecting k data points."""
    centroids_idx = random.sample(range(len(data)), k)
    centroids = [data[idx] for idx in centroids_idx]  # Corrected line
    return centroids

def kmeans(data, k, max_iterations=100):
    """K Means clustering algorithm."""
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for point in data:
            closest_centroid_index = min(
                range(k),
                key=lambda i: euclidean_distance(point, centroids[i])
            )
            clusters[closest_centroid_index].append(point)
        new_centroids = [
            [sum(dim) / len(cluster) for dim in zip(*cluster)]
            if cluster else random.choice(data)
            for cluster in clusters
        ]
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, clusters

if __name__ == "__main__":
    data = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(100)]
    k = 3
    profiler = cProfile.Profile()
    profiler.enable()
    centroids, clusters = kmeans(data, k)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('time')
    stats.print_stats()

    print("Centroids:", centroids)
