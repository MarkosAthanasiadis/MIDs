# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

import torch
import numpy as np
import utils.train_test_model as md
import torch.nn as nn


def computer_local(eigenvectors, modelinfo, nsub, nvalues, model_flag):
    """
    Adjusts the direction of eigenvectors such that they consistently point towards one class.

    This function takes a set of eigenvectors and adjusts their direction based on the 
    classification of the vectors using a trained model. The eigenvectors are flipped
    to align with the predicted class, ensuring all eigenvectors point towards a single class.

    Parameters:
        eigenvectors (numpy.ndarray): The identified eigenvectors to be adjusted.
        modelinfo (str): Path prefix for model files.
        nsub (int): Subsampling index for identifying the model.
        nvalues (list): List of values indicating the number of features.
        model_flag (str): Flag indicating the type of model ('linear' or 'non-linear').

    Returns:
        numpy.ndarray: The direction-adjusted eigenvectors.
    """
    # Determine how many features are needed
    needed_features = len([1 for x in nvalues if x != 1])
    
    # Concatenate eigenvectors from all adversarial patterns
    eigenvectors = np.concatenate(eigenvectors, axis=0)
    
    # Load the model corresponding to the subsample index
    model_name = f'{modelinfo}Model_Sub{nsub}.pt'
    inference_model = md.LinearNet(needed_features) if model_flag == 'linear' else md.NonLinearNet(needed_features)
    inference_model.load_state_dict(torch.load(model_name))
    inference_model.eval()
    
    # Convert eigenvectors to torch tensors for inference
    cluster_test = torch.tensor(eigenvectors).float()
    
    # Predict labels for the eigenvectors
    cluster_test_label = inference_model.predict(cluster_test).numpy()
    
    # Flip the sign in any of the eigenvectors for which the infered label belongs to class 0 
    flip_eigenvectors = []
    for neig in range(eigenvectors.shape[0]):
        eig_i = eigenvectors[neig,:]
        eig_i = np.reshape(eig_i,(1,len(eig_i)))
        if cluster_test_label[neig]==0:
            eig_i = eig_i * (-1)
        flip_eigenvectors.append(eig_i)
    eigenvectors = np.concatenate(flip_eigenvectors,axis=0)
    
    return eigenvectors



def computer_global(meaned_clusters, x_data, y_labels, dim_i, lr, epochs, modelinfo, model_flag):
    """
    Adjusts the direction of averaged patterns by training a model on the entire dataset.

    This function trains a model using the full dataset and adjusts the direction of
    the averaged patterns to point towards one of the classes based on the model's 
    classification. This serves as a failsafe when local adjustments are insufficient.

    Parameters:
        meaned_clusters (numpy.ndarray): Averaged patterns to be adjusted.
        x_data (numpy.ndarray): Feature data used for training the model.
        y_labels (numpy.ndarray): Corresponding labels for the training data.
        dim_i (int): The dimensionality of the input data.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        modelinfo (str): Path prefix for saving the trained model.
        model_flag (str): Flag indicating the type of model ('linear' or 'non-linear').

    Returns:
        numpy.ndarray: The direction-adjusted averaged patterns.
    """
    # Convert data and labels to torch tensors
    X_train = torch.tensor(x_data).float()
    y_train = torch.tensor(y_labels).long()
    
    # Initialize the model based on the model flag
    model = md.LinearNet(dim_i) if model_flag == 'linear' else md.NonLinearNet(dim_i)
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    
    # Save the trained model
    model_name = f'{modelinfo}Model_Global.pt'
    torch.save(model.state_dict(), model_name)
    
    # Predict labels for the averaged clusters
    cluster_test = torch.tensor(meaned_clusters).float()
    cluster_test_label = model.predict(cluster_test).numpy()
    
    # Flip the sign in any of the eigenvectors for which the infered label belongs to class 0 
    flip_meaned_clusters = []
    for neig in range(meaned_clusters.shape[0]):
        vec_i = meaned_clusters[neig,:]
        vec_i = np.reshape(vec_i,(1,len(vec_i)))
        if cluster_test_label[neig]==0:
            vec_i = vec_i * (-1)
        flip_meaned_clusters.append(vec_i)
    meaned_clusters = np.concatenate(flip_meaned_clusters,axis=0)  
    
    return meaned_clusters


