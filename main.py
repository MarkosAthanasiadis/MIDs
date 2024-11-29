# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

"""
Main Adversarial Decoder Logic
---------------------------------
This module implements the main adversarial decoder logic that orchestrates all helper functions 
required for adversarial decoding analysis.
It handles data preparation, model training, adversarial attacks, and computation/analysis of results.

"""

import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import pickle
from utils import train_test_model as md
from utils import neighborhood_computation as neigh
from utils import eigendecomposition as eigde
from utils import inference as infer
from utils import clustering as cl
from utils import unification as unt
from utils import gravity_center as grav
from utils import projections as proj
from utils import relevance as rp



def main(att_res):
    """
    Main adversarial decoder function that calls all helper functions.

    Args:
        att_res (dict): A dictionary containing configuration parameters and data.

    Returns:
        None: Results dictionary including decoding CCRs, including clustering, 
        consistency metrics, MIDs and projection vectors. Updates parameters dictionary
        in place.
        
    """

    # Unpack classifier parameters
    model_save_path   = att_res['parameters']['model_attack_savepaths'][0]
    attack_save_path  = att_res['parameters']['model_attack_savepaths'][1]
    num_subsamplings  = att_res['parameters']['nsubsamplings']
    model_type        = att_res['parameters']['model_flag']
    learning_rate     = att_res['parameters']['lr_epochs'][0]
    num_epochs        = att_res['parameters']['lr_epochs'][1]
    ring_range_start  = att_res['parameters']['ring_domain'][0]
    ring_range_stop   = att_res['parameters']['ring_domain'][1]
    min_samples       = att_res['parameters']['min_samples_epsilon'][0]
    epsilon_threshold = att_res['parameters']['min_samples_epsilon'][1]
    specified_cores   = att_res['computation_cores']
    data_features     = att_res['original_data']
    data_labels       = att_res['original_labels']
    feature_dimension = data_features.shape[1]

    # Determine available computational cores
    subsampling_runs = range(1, num_subsamplings + 1)
    available_cores = multiprocessing.cpu_count()
    if specified_cores != -1:
        if specified_cores <= available_cores:
            available_cores = specified_cores
        else:
            raise ValueError('Not enough physical cores available. Try again with a different amount.')

    # Parallelized training and testing
    subsampling_results = Parallel(n_jobs=available_cores, timeout=None)(
        delayed(md.train_test_model)(
            np.copy(data_features), 
            np.copy(data_labels), 
            feature_dimension, 
            learning_rate, 
            num_epochs, 
            subsample_idx, 
            model_save_path, 
            model_type
        ) for subsample_idx in subsampling_runs
    )

    # Process the results from parallel training
    for idx, result in enumerate(subsampling_results):
        subsampling_key = f'Subsampling_{idx + 1}'
        att_res[subsampling_key] = result


    # Save intermediate results
    results_file_name = f"{attack_save_path}analysis_results"
    with open(results_file_name, 'wb') as file:
        pickle.dump(att_res, file)

    # Process adversarial data for each subsampling
    for subsample_idx in range(1, num_subsamplings + 1):
        subsampling_key = f'Subsampling_{subsample_idx}'
        adversarial_samples = att_res[subsampling_key]['adversarial_data']
        useless_features_count = att_res[subsampling_key]['useless_features']

        # Compute local neighborhood
        neighborhood_indices = neigh.computer(
            adversarial_samples, 
            ring_range_start, 
            ring_range_stop, 
            feature_dimension, 
            available_cores
        )

        # Compute weight vectors and indices
        eigenvectors, valid_indices = eigde.computer(neighborhood_indices, adversarial_samples, available_cores)

        # Ensure local normal vectors are computed
        if eigenvectors:
            # Adjust eigenvector direction using inferred labels
            adjusted_vectors = infer.computer_local(
                eigenvectors, 
                model_save_path, 
                subsample_idx, 
                useless_features_count, 
                model_type
            )

            # Perform local clustering
            local_clusters = cl.computer_local(adjusted_vectors, min_samples, epsilon_threshold, useless_features_count)

            # Store clustering results
            att_res[subsampling_key]['local_clustering_data'] = local_clusters
            att_res[subsampling_key]['valid_indices'] = valid_indices

    # Save updated results
    with open(results_file_name, 'wb') as file:
        pickle.dump(att_res, file)

    # Combine results from all subsamplings
    cluster_means, cluster_counts, subsample_ids, cluster_indices = unt.computer(num_subsamplings, att_res)

    # Train global vectors and flip labels
    global_vectors = infer.computer_global(
        cluster_means, data_features, data_labels, feature_dimension, 
        learning_rate, num_epochs, model_save_path, model_type
    )

    # Perform global clustering
    final_global_vectors, final_counts, consistency_score, global_cluster_indices = cl.computer_global(
        global_vectors, cluster_counts, num_subsamplings, subsample_ids, min_samples, epsilon_threshold
    )

    # Update global attack results
    att_res['MIDs'] = final_global_vectors
    att_res['MID_indices'] = global_cluster_indices
    att_res['consistency'] = consistency_score
    att_res['amount'] = final_counts

    # Save final results
    with open(results_file_name, 'wb') as file:
        pickle.dump(att_res, file)

    # Compute gravity centers
    gravity_centers = grav.computer(att_res, global_cluster_indices, subsample_ids, cluster_indices)
    att_res['gravity_centers'] = gravity_centers

    # Compute projection vectors
    projection_vectors = proj.computer(data_features, final_global_vectors, gravity_centers)
    att_res['projections'] = projection_vectors

    # Save final results again
    with open(results_file_name, 'wb') as file:
        pickle.dump(att_res, file)

    # Initialize a dictionary to hold the relevance results
    att_res['RPs'] = {}

    # Loop over the identified MIDs that exceed the consistency threshold
    for nvec in range(len(final_global_vectors)):

        # Grab the projection and the global vector
        global_vector_i = final_global_vectors[nvec]
        center_i = gravity_centers[nvec]
        projection_i = projection_vectors[nvec]

        # Corrected  the data
        corrected_data = []
        x_data_i = np.copy(data_features)
        for i in range(x_data_i.shape[0]):
            data_i = x_data_i[i,:]
            data_i = np.reshape(data_i,(1,len(data_i)))
            data_i = data_i - center_i
            data_i = data_i /np.linalg.norm(data_i)
            corrected_data.append(data_i)
        corrected_data = np.concatenate(corrected_data,axis=0)

        # Shuffle the indices of the features multiple times
        nshuffles, perms = rp.shuffle(feature_dimension)
        if nshuffles > 10:
            # Compute shuffled projections in parallel
            runs = range(0, nshuffles)
            shuffled_projections = Parallel(n_jobs=specified_cores, timeout=None)(
                delayed(rp.dot)(corrected_data, perms[nsh], global_vector_i) for nsh in runs
            )

            # Identify relevant and irrelevant datapoints based on the shuffled projections
            relevant_indices, irrelevant_indices, relevant_upper_indices, relevant_lower_indices = rp.find(
                shuffled_projections, runs, projection_i
            )

        # Initialize an entry per MID and store the indices found to be relevant
        att_res['RPs'][f'MID_{nvec + 1}'] = {
            'relevant_indices': relevant_indices,
            'irrelevant_indices': irrelevant_indices,
            'upper_percentile_relevant_indices': relevant_upper_indices,
            'lower_percentile_relevant_indices': relevant_lower_indices
        }

    # Save final results again
    with open(results_file_name, 'wb') as file:
        pickle.dump(att_res, file)

    return


