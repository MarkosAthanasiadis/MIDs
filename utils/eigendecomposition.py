# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""
from joblib import Parallel, delayed
import mkl
import numpy as np


def eigendecomposition(cov_mat):
    """
    eigendecomposition
    ______________________
    
    This function performs eigendecomposition on the covariance matrix 
    to identify the eigenvector corresponding to the smallest eigenvalue.
    
    Parameters:  
        cov_mat (numpy.ndarray): A covariance matrix.

    Returns: 
        numpy.ndarray: The eigenvector corresponding to the minimum eigenvalue.
    """
    # Perform eigendecomposition to obtain eigenvalues and eigenvectors
    values, vectors = np.linalg.eig(cov_mat)
    
    # Get the index of the minimum eigenvalue
    min_value_idx = np.argmin(values)
    
    # Extract the eigenvector corresponding to the minimum eigenvalue
    min_vector = vectors[:, min_value_idx]
    
    # Extract the real part of the eigenvector and reshape it for further analysis
    min_vector = np.real(min_vector)
    min_vector = np.reshape(min_vector, (1, len(min_vector)))
    
    return min_vector




def computer(ind, adversarial_data, num_cores):
    """
    computer
    ______________________
    
    This function computes the covariance matrix for each adversarial pattern 
    and performs eigendecomposition to find the eigenvector corresponding 
    to the smallest eigenvalue. This eigenvector is the one that is most normal 
    to the decision boundary of the trained model, as it represents the direction 
    with the lowest data variance.
    
    Parameters:  
        ind (list): List of indices for each adversarial pattern's neighbors within the ring domain.
        adversarial_data (numpy.ndarray): Array of adversarial patterns.
        num_cores (int): Number of computational cores to be used.

    Returns: 
        list: A list of eigenvectors corresponding to each adversarial pattern.
        list: A list of indices for patterns where an eigenvector could be computed.
    """
    mkl.set_num_threads(num_cores)
    
    # Initialize dictionary to store covariance matrices and list for useful indices
    Amat = {}
    indexmat = 0
    useful_indices = []
    
    # Compute covariance matrix for each datapoint where neighbors exist
    for n_data in range(adversarial_data.shape[0]):
        if len(ind[n_data]) != 0:  # Ensure neighbors exist in the ring domain
            # Create a unique key for each datapoint's covariance matrix
            covariance_key = f'Covariance_Datapoint_{indexmat + 1}'
            indexmat += 1
            
            # Compute the difference matrix (di - dj) for all neighbors of the current datapoint
            dij_matrix = adversarial_data[n_data, :] - adversarial_data[ind[n_data], :]
            
            # Compute the covariance matrix: dij.T * dij
            covariance_matrix = np.matmul(np.transpose(dij_matrix), dij_matrix)
            
            # Store the covariance matrix and the index of the datapoint
            Amat[covariance_key] = covariance_matrix
            useful_indices.append(n_data)
    
    # Initialize the list to hold the eigenvectors
    eigenvectors = []
    
    if Amat:
        # Perform parallel eigendecomposition for all datapoints with available covariance matrices
        runs = range(len(Amat))
        eigenvectors = Parallel(n_jobs=num_cores)(
            delayed(eigendecomposition)(Amat[f'Covariance_Datapoint_{n_data + 1}']) for n_data in runs
        )
    
    return eigenvectors, useful_indices


