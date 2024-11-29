# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

import numpy as np

def computer(att_res, global_clusters_indices, subs_ids, meaned_clusters_indices):
    """
    Computes the gravity center (mean center) for each of the identified global vectors/MIPs.

    Parameters:
        att_res (dict): The adversarial data from each subsampling.
        global_clusters_indices (list): Indices of local normals (vectors) contributing to each global vector.
        subs_ids (list): Subsampling IDs for each local normal vector.
        meaned_clusters_indices (list): Indices of the eigenvectors that contributed to each local normal vector.

    Returns:
        list: Gravity centers for each global vector/MIP.
    """

    # List to hold the gravity centers for each global vector
    gravity_centers = []

    # Loop over each global vector
    for nvec, global_indices_i in enumerate(global_clusters_indices):

        # List to hold adversarial data points contributing to this global vector
        adversarials = []

        # Loop over the local normals contributing to this global vector
        for nloc in global_indices_i:

            # Get the subsampling ID and key
            sub_id = subs_ids[nloc]
            subsampling_key = f'Subsampling_{sub_id}'

            # Get the indices of eigenvectors creating the local normal vector
            eig_inds = meaned_clusters_indices[nloc]

            # Get the relevant adversarial data for this subsampling
            usefull_inds = att_res[subsampling_key]['valid_indices']
            nvalues = att_res[subsampling_key]['useless_features']
            deleted_features = [0 if x == 1 else 1 for x in nvalues]
            adversarials_i = att_res[subsampling_key]['adversarial_data']

            # Fill in the missing dimensions (replace deleted features with zeros)
            filled_adversarials = []
            indexer = 0
            for ndim in range(len(nvalues)):
                if deleted_features[ndim] == 1:
                    filled_adversarials_i = adversarials_i[:, indexer].reshape(-1, 1)
                    indexer += 1
                else:
                    filled_adversarials_i = np.zeros((adversarials_i.shape[0], 1))
                filled_adversarials.append(filled_adversarials_i)
            adversarials_i = np.concatenate(filled_adversarials, axis=1)

            # Loop over the eigenvectors that created the local normal vector
            for neig in eig_inds:
                # Get the center ID for each eigenvector and retrieve the center datapoint
                center_id = usefull_inds[neig]
                center_i = adversarials_i[center_id, :].reshape(1, -1)

                # Store the adversarial (center) data for this global vector
                adversarials.append(center_i)

        # Concatenate all adversarial points and compute the center of gravity (mean)
        adversarials = np.concatenate(adversarials, axis=0)
        center_gravity_i = np.mean(adversarials, axis=0).reshape(1, -1)

        # Store the gravity center for this global vector
        gravity_centers.append(center_gravity_i)

    return gravity_centers
