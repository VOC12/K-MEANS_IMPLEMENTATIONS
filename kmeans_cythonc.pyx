# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# Euclidean distance calculation optimized with Cython
cdef double euclidean_distance(double[:] p1, double[:] p2) nogil:
    cdef double dist = 0
    cdef double diff
    cdef int i
    for i in range(p1.shape[0]):
        diff = p1[i] - p2[i]
        dist += diff * diff
    return sqrt(dist)

# Main K-Means clustering algorithm function
cpdef kmeans_cython(np.ndarray[np.double_t, ndim=2] data, int k, int max_iterations=100):
    cdef int num_points = data.shape[0]
    cdef int num_dims = data.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] centroids = np.zeros((k, num_dims), dtype=np.double)
    cdef np.ndarray[np.int_t, ndim=1] nearest_centroid_indices = np.zeros(num_points, dtype=int)
    cdef np.ndarray[np.double_t, ndim=1] distances = np.zeros(k, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] new_centroids
    cdef list clusters = [np.empty((0, num_dims), dtype=np.double) for _ in range(k)]
    
    # Initialize centroids randomly
    for i in range(k):
        centroids[i] = data[np.random.randint(0, num_points)]

    # K-Means iteration
    for _ in range(max_iterations):
        for i in range(k):
            clusters[i] = np.empty((0, num_dims), dtype=np.double)

        # Assign points to the nearest centroid
        for i in range(num_points):
            for j in range(k):
                distances[j] = euclidean_distance(data[i], centroids[j])
            nearest_centroid_indices[i] = np.argmin(distances)
            clusters[nearest_centroid_indices[i]] = np.vstack((clusters[nearest_centroid_indices[i]], data[i]))

        # Update centroids
        for i in range(k):
            if len(clusters[i]) > 0:
                centroids[i] = np.mean(clusters[i], axis=0)
                
    return centroids, clusters
