# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

import numpy as np

def computer(subsamplings, att_res):
    """
    Averages over all patterns within each cluster, then combines all averaged patterns into a single array.
    
    Parameters:
        subsamplings (int): The number of subsamplings performed.
        att_res (dict): The clustering results for each subsampling. Should contain the local clustering data.
        
    Returns:
        tuple:
            - meaned_clusters (numpy.ndarray): Array of averaged (and possibly flipped) cluster vectors.
            - amount (list): A list indicating how many eigenvectors belonged to each averaged pattern.
            - subs_ids (list): A list of subsampling IDs for each averaged pattern.
            - meaned_clusters_indices (list): Indices of the eigenvectors in each cluster.
    """
    amount = []
    meaned_clusters = []
    meaned_clusters_indices = []
    meaned_clusters_normed = []
    subs_ids = []

    # Loop over each subsampling
    for n_subs in range(1, subsamplings + 1):
        subsampling_key = f'Subsampling_{n_subs}'

        # Check if this subsampling contains clustering results
        if 'local_clustering_data' in att_res.get(subsampling_key, {}):
            local_clusters_data = att_res[subsampling_key]['local_clustering_data']

            # Loop over each cluster in the subsampling
            for cluster_key, cluster_data in local_clusters_data.items():
                cluster_i = cluster_data['data']
                cluster_i_indices = cluster_data['indices']

                # Store the number of eigenvectors in this cluster
                amount.append(cluster_i.shape[0])
                subs_ids.extend([n_subs] * cluster_i.shape[0])

                # Calculate the mean resultant vector for the cluster
                cluster_mean = np.mean(cluster_i, axis=0).reshape(1, -1)

                # Store the mean cluster vector and its normalized version
                meaned_clusters.append(cluster_mean)
                meaned_clusters_indices.append(cluster_i_indices)
                meaned_clusters_normed.append(cluster_mean / np.linalg.norm(cluster_mean))

    # Concatenate all meaned clusters and their normalized versions
    meaned_clusters = np.concatenate(meaned_clusters, axis=0)
    meaned_clusters_normed = np.concatenate(meaned_clusters_normed, axis=0)

    # Initialize a flag array to track which vectors have been flipped
    flip_flags = np.zeros(meaned_clusters.shape[0], dtype=bool)

    # Flip vectors that are close to being antipodal
    for nvec in range(meaned_clusters_normed.shape[0]):
        if not flip_flags[nvec]:
            vec_i = meaned_clusters_normed[nvec, :].reshape(1, -1)
            dots = np.dot(vec_i, meaned_clusters_normed.T)

            # Find vectors with a dot product close to -1 (antipodal vectors)
            inds = [ind for ind, x in enumerate(dots[0, :]) if x < -0.95 and ind > nvec]
            unflipped_inds = [ind for ind in inds if not flip_flags[ind]]

            # Flip the sign of the vectors that are antipodal
            for ind in unflipped_inds:
                flip_flags[ind] = True

    # Apply the flips and normalize the vectors
    flipped_clusters = []
    for neig in range(meaned_clusters.shape[0]):
        vec_i = meaned_clusters[neig, :]
        if flip_flags[neig]:
            vec_i = -vec_i
        flipped_clusters.append(vec_i.reshape(1, -1))

    meaned_clusters = np.concatenate(flipped_clusters, axis=0)

    return meaned_clusters, amount, subs_ids, meaned_clusters_indices

