import tensorflow as tf
import numpy as np
from kmeansfunctions import choose_random_centroids, assign_to_nearest, update_centroids

def k_means_1_iteration(sample_values, centroid_values,n_clusters):
    samples = tf.constant(sample_values)
    if len(centroid_values):
        updated_centroids = choose_random_centroids(samples, n_clusters)
    else:
        updated_centroids = tf.constant(centroid_values)
    nearest_indices,j_cost = assign_to_nearest(samples, updated_centroids)
    updated_centroids = update_centroids(samples, nearest_indices, n_clusters)
    nearest_indices,j_cost = assign_to_nearest(samples, updated_centroids)
    j_cost = tf.reduce_sum(j_cost)
    with tf.compat.v1.Session() as session:
        j_cost_value, updated_centroid_values = session.run((j_cost,updated_centroids))
    return j_cost_value, updated_centroid_values

def kmeans_general_func2(sample_values, n_clusters, n_iterations):
    centroid_values = [0]
    j_costs = np.zeros(n_iterations)
    for i in range(n_iterations):
        j_cost_value,centroid_values = k_means_1_iteration(sample_values,centroid_values,n_clusters)
        j_costs[i] = j_cost_value
        if i==0:
            best_j = j_costs[i]
            best_centroid_values = centroid_values
        elif j_costs[i]<=best_j:
            best_j = j_costs[i]
            best_centroid_values = centroid_values
    return j_costs,best_centroid_values,best_j
