# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

from sklearn.cluster import DBSCAN
import numpy as np

def computer_local(eigenvectors, min_samples, epsilon, nvalues):
    """
    Clusters the identified eigenvectors based on their direction in space using DBSCAN.

    Parameters:
        eigenvectors (numpy.ndarray): The identified eigenvectors to be clustered.
        min_samples (int): Minimum number of samples required for a cluster.
        epsilon (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        nvalues (list): List indicating which features are kept (1) or deleted (0).

    Returns:
        dict: A dictionary containing the clustered data and the indices of the eigenvectors for each cluster.
    """
    # Determine deleted features and original dimensions
    deleted_features = [0 if x == 1 else 1 for x in nvalues]
    dimensions = len(nvalues)

    # Adjust min_samples to be a percentage of the dataset size
    min_samples = int((min_samples / 100) * eigenvectors.shape[0])

    # DBSCAN clustering loop with adaptive epsilon
    while True:
        clustering = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean', n_jobs=-1).fit(eigenvectors)
        predictions = clustering.labels_

        # Check if at least one cluster was found
        n_clusters = max(predictions) + 1
        if n_clusters >= 1:
            break
        else:
            epsilon += 0.1  # Increase epsilon if no clusters found

    # Prepare the dictionary to store the clusters data
    local_clusters_data = {}

    # Iterate over all identified clusters
    for nid in range(n_clusters):
        # Get indices and local normals for each cluster
        indices = [ind for ind, x in enumerate(predictions) if x == nid]
        local_normals = eigenvectors[indices, :]

        # Restore missing dimensions
        dummy_local_normals = []
        indexer = 0
        for ndim in range(dimensions):
            if deleted_features[ndim] == 1:
                dummy_local_normals_i = np.reshape(local_normals[:, indexer], (len(local_normals), 1))
                indexer += 1
            else:
                dummy_local_normals_i = np.zeros((local_normals.shape[0], 1))
            dummy_local_normals.append(dummy_local_normals_i)

        local_normals = np.concatenate(dummy_local_normals, axis=1)

        # Store the cluster data in the dictionary
        cluster_key = f'Cluster_{nid + 1}'
        local_clusters_data[cluster_key] = {'data': local_normals, 'indices': indices}

    return local_clusters_data


def computer_global(meaned_clusters, local_amount, subsamplings, subs_ids, min_samples, epsilon):
    """
    Clusters the averaged patterns from subsamplings using DBSCAN.

    Parameters:
        meaned_clusters (numpy.ndarray): Averaged patterns to be clustered.
        local_amount (list): Metric indicating the "amount" of each pattern.
        subsamplings (list): List of subsamplings used for averaging the patterns.
        subs_ids (list): List of subsample IDs corresponding to the data points.
        min_samples (int): Minimum number of samples required for a cluster.
        epsilon (float): The maximum distance between two samples for them to be considered as in the same neighborhood.

    Returns:
        tuple: 
            - global_clusters_data (list): Clusters' global normalized vectors.
            - amount (list): The "amount" metric for each cluster.
            - consistency (list): The consistency of each cluster based on the number of subsamplings contributing to it.
            - global_clusters_indices (list): Indices of the data points in each cluster.
    """
    # Adjust min_samples to be a percentage of the dataset size
    min_samples = int((min_samples / 100) * meaned_clusters.shape[0])

    # DBSCAN clustering loop with adaptive epsilon
    while True:
        clustering = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean', n_jobs=-1).fit(meaned_clusters)
        predictions = clustering.labels_

        # Check if at least one cluster was found
        n_clusters = max(predictions) + 1
        if n_clusters >= 1:
            break
        else:
            epsilon += 0.1  # Increase epsilon if no clusters found

    # Initialize variables to hold cluster data and metrics
    global_clusters_data = []
    global_clusters_indices = []
    amount = []
    consistency = []

    # Loop over all identified clusters
    for nid in range(n_clusters):
        # Get indices for the current cluster
        indices = [ind for ind, x in enumerate(predictions) if x == nid]

        # Calculate the amount and consistency for the current cluster
        amount_i = np.sum([local_amount[x] for x in indices]) / np.sum(local_amount)
        amount.append(np.round(100 * amount_i, 2))

        consistency_i = len(np.unique([subs_ids[x] for x in indices])) / len(np.unique(subs_ids))
        consistency_i = min(consistency_i, 1)  # Ensure consistency is capped at 100%
        consistency.append(np.round(100 * consistency_i, 2))

        # Calculate the global vector for this cluster
        global_normals = np.mean(meaned_clusters[indices, :], axis=0)
        global_normals = np.reshape(global_normals, (1, len(global_normals)))

        # Normalize the global vector to unit length
        global_normals /= np.linalg.norm(global_normals)

        # Store the cluster data
        global_clusters_data.append(global_normals)
        global_clusters_indices.append(indices)

    # Sort clusters by consistency (descending)
    sorting_indices = np.flip(np.argsort(consistency))
    amount = [amount[i] for i in sorting_indices]
    consistency = [consistency[i] for i in sorting_indices]
    global_clusters_data = [global_clusters_data[i] for i in sorting_indices]

    return global_clusters_data, amount, consistency, global_clusters_indices



