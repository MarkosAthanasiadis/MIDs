# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

import itertools
import numpy as np


def shuffle(dimensions):
    """
    Generate shuffled permutations of dataset dimensions.

    Parameters:
        dimensions (int): The dimensionality of the dataset.

    Returns:
        tuple: A tuple containing:
            - int: The total number of unique shuffles.
            - list: A list of shuffled indices per permutation.
    """
    
    # Initialize an array with dimension indices
    shuffle_inds = np.arange(dimensions)
    
    # Determine the number of shuffles (up to 1000 or factorial of dimensions)
    nshuffles = 1000
    if dimensions <= 6:
        max_permutations = np.math.factorial(dimensions)
        nshuffles = min(1000, max_permutations)

    perms = []

    if nshuffles != np.math.factorial(dimensions):
        # Generate random unique permutations until required number is reached
        while len(perms) < nshuffles:
            perm_i = list(np.random.permutation(shuffle_inds))
            if perm_i not in perms:
                perms.append(perm_i)
    else:
        # Generate all permutations if dimensions are small
        perms = list(itertools.permutations(shuffle_inds))
        perms = perms[1:]  # Exclude the unshuffled case

    return nshuffles, perms


def dot(corrected_data, shuffler, pattern_i):
    """
    Compute the projection vector for shuffled data.

    Parameters:
        corrected_data (ndarray): Gravity center adjusted dataset.
        shuffler (list): Indices to shuffle the data.
        pattern_i (ndarray): The global vector (or MIP) to project onto.

    Returns:
        ndarray: A projection vector for the shuffled data.
    """
    # Shuffle the corrected data based on the shuffler indices
    shuffled_data = corrected_data[:, shuffler]

    # Compute the projection using the dot product
    sh_projection = np.dot(shuffled_data, pattern_i.T)
    sh_projection = np.reshape(sh_projection, (1, len(sh_projection)))

    return sh_projection


def find(sh_projection, runs, projection_i):
    """
    Identify relevant patterns for each global vector (MIP).

    Parameters:
        sh_projection (list of ndarray): Shuffled projections.
        runs (range): Range of shuffle iterations.
        projection_i (ndarray): Real projection vector.
        up (int, optional): Upper percentile for outlier detection (default=97.5).
        down (int, optional): Lower percentile for outlier detection (default=2.5).

    Returns:
        tuple: A tuple containing lists of indices for:
            - relevant data points
            - irrelevant data points
            - positively relevant data points
            - negatively relevant data points
    """
    # Combine projections from all runs into a single distribution
    proj_dist = np.concatenate(sh_projection, axis=0)
    proj_dist = proj_dist.flatten()

    # Determine outlier thresholds based on percentiles
    perc5 = np.percentile(proj_dist, 2.5)
    perc95 = np.percentile(proj_dist, 97.5)

    # Classify indices based on thresholds
    relevant_indices = [i for i, x in enumerate(projection_i[:, 0]) if x > perc95 or x < perc5]
    irrelevant_indices = [i for i, x in enumerate(projection_i[:, 0]) if perc5 <= x <= perc95]
    relevant_upper_indices = [i for i, x in enumerate(projection_i[:, 0]) if x > perc95]
    relevant_lower_indices = [i for i, x in enumerate(projection_i[:, 0]) if x < perc5]

    return relevant_indices, irrelevant_indices, relevant_upper_indices, relevant_lower_indices
