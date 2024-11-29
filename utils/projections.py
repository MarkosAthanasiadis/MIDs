# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

import numpy as np

def computer(x_data, global_vectors, gravity_centers):
    """
    Corrects the input dataset based on the gravity center for each identified global vector/MIP.
    Then the projection of the MIP with the corrected dataset is computed.

    Parameters:
        x_data (numpy.ndarray): The dataset to be corrected and projected.
        global_vectors (list): The identified global vectors (MIPs).
        gravity_centers (list): The gravity centers corresponding to each global vector.

    Returns:
        list: The projections of the global vectors onto the corrected dataset.
    """
    
    # Initialize the list to hold the projection vectors
    q_vectors = []

    # Loop over each global vector
    for nvec in range(len(global_vectors)):
        
        global_vector_i = global_vectors[nvec]
        center_i = gravity_centers[nvec]
        
        # Corrected  the data
        corrected_data = []
        x_data_i = np.copy(x_data)
        for i in range(x_data_i.shape[0]):
            data_i = x_data_i[i,:]
            data_i = np.reshape(data_i,(1,len(data_i)))
            data_i = data_i - center_i
            data_i = data_i /np.linalg.norm(data_i)
            corrected_data.append(data_i)
        corrected_data = np.concatenate(corrected_data,axis=0)
        
        # Compute the projection of the corrected data onto the global vector
        projection_i = np.dot(corrected_data, global_vector_i.T)
    
        # Store the projection
        q_vectors.append(projection_i)
    
    return q_vectors




