# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

import numpy as np
import pickle

def load_and_prepare_data(mainpath, dataname):
    """
    Load and prepare a dataset for a permutation test.

    This function loads a dataset stored as a Python dictionary from a file, 
    extracts the data and labels, and prepares them for analysis. Specifically, 
    it converts the data into a unified format, and adjusts the labels to be binary 
    (0 or 1) for model training.

    Parameters:
    -----------
    mainpath : str
        The path to the directory containing the dataset file.
    
    dataname : str
        The name of the dataset file (with extension). The file should contain a 
        Python dictionary with the following structure:
        - 'data': A list of numpy arrays (each array has shape (nxD)).
        - 'labels': A list of integer values corresponding to each trial.

    Returns:
    --------
    x_data : np.ndarray
        The concatenated data array with shape (N, D), where N is the total number 
        of patterns and D is the number of dimensions.
    
    y_data : np.ndarray
        The concatenated labels array with shape (N,), where N is the total number 
        of patterns.
    
    trial_labels : list
        A list of the trial labels (either 0 or 1) for each trial.
    
    trial_contributions : list
        A list of integers representing the number of patterns in each trial (i.e., 
        the contribution of each trial).
    
    """
    
    # Load the dataset from the pickle file
    with open(mainpath + dataname, 'rb') as f:
        dataset = pickle.load(f)

    # Extract the data and labels
    data = dataset['data']
    labels = dataset['labels']
    
    # Initialize lists to hold the processed data
    x_data, y_data, trial_labels, trial_contributions = [], [], [], []

    # Loop over each trial in the dataset
    n_trials = len(data)
    for trial_idx in range(n_trials):
        trial_data = data[trial_idx]
        trial_label = labels[trial_idx]
        
        # Get the number of patterns (contribution) in this trial
        contribution = trial_data.shape[0]
        
        # Create labels for this trial
        trial_label_array = np.full((contribution,), trial_label)
        
        # Append the trial data, labels, and contributions to the lists
        x_data.append(trial_data)
        y_data.append(trial_label_array)
        trial_contributions.append(contribution)
        trial_labels.append(trial_label)
    
    # Convert the lists into numpy arrays
    x_data = np.concatenate(x_data, axis=0)
    y_data = np.concatenate(y_data, axis=0)

    # Convert labels to binary (0 or 1)
    unique_labels = np.unique(y_data)
    for i, label in enumerate(y_data):
        y_data[i] = 0 if label == unique_labels[0] else 1
    for i, label in enumerate(trial_labels):
        trial_labels[i] = 0 if label == unique_labels[0] else 1
    
    return x_data, y_data, trial_labels, trial_contributions















