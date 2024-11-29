# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
from joblib import Parallel, delayed
import mkl

def sorter(indexer, distancer, r_start, r_stop):
    """
    sorter
    ______________________
    
    This function identifies the neighbors that are included within a ring-shaped domain 
    around each of the adversarial patterns based on the distance criteria.
    
    Parameters:  
        indexer (array-like): Indices of all other patterns.
        distancer (array-like): The distances of all other patterns from each adversarial pattern.
        r_start (float): The start distance for the ring-shaped domain.
        r_stop (float): The end distance for the ring-shaped domain.

    Returns: 
        list: Indices of neighbors that belong within the specified ring-shaped domain.
    """
    return [index for distance, index in zip(distancer, indexer) if r_start < distance < r_stop]




def computer(adversarial_data, ring_start, ring_stop, dimensions, num_cores):
    """
    computer
    ______________________
    
    This function identifies the closest neighbors of each adversarial pattern 
    and forms a ring-shaped local neighborhood around each pattern.

    Parameters:  
        adversarial_data (numpy.ndarray): Array of adversarial patterns.
        ring_start (float): The starting distance for the ring-shaped domain.
        ring_stop (float): The ending distance for the ring-shaped domain.
        dimensions (int): The dimensionality of the data.
        num_cores (int): Number of computational cores to be used.

    Returns: 
        list: List of indices for each adversarial pattern that belongs within the ring domain.
    """
    # Fit NearestNeighbors model
    neigh = NearestNeighbors(n_neighbors=adversarial_data.shape[0], n_jobs=-1)
    neigh.fit(adversarial_data)
    dist, ind = neigh.kneighbors(adversarial_data, adversarial_data.shape[0], return_distance=True)
    
    # Calculate the min and max distances for each data point
    min_dist = [next((value for index, value in enumerate(dist_i) if value != 0), 0) for dist_i in dist]
    max_dist = [dist_i[-1] for dist_i in dist]
    
    # Reshape min and max distances into column vectors
    min_dist = np.reshape(min_dist, (len(min_dist), 1))
    max_dist = np.reshape(max_dist, (len(max_dist), 1))
    
    # Compute the delta between min and max distance for each data point
    delta_dist = max_dist - min_dist
    
    # Compute the start and stop values for the ring-shaped local neighborhood
    r_start = min_dist + (delta_dist * ring_start) / 100
    r_stop = min_dist + (delta_dist * ring_stop) / 100
    
    # Set the number of threads to be used for parallel processing
    mkl.set_num_threads(num_cores)
    
    # Parallelize the identification of neighbors within the ring domain
    runs = range(adversarial_data.shape[0])
    keeper = Parallel(n_jobs=num_cores)(
        delayed(sorter)(ind[n_data], dist[n_data], r_start[n_data], r_stop[n_data]) for n_data in runs
    )

    return keeper

