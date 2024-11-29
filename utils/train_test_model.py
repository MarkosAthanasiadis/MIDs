# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.attack_foolbox as fool

def subsample_data(label_indices, train_target):
    """
    Subsamples the data by shuffling the indices of each label, 
    and splits them into training and testing sets.

    Args:
        label_indices (list): Indices of data points corresponding to a label.
        train_target (int): The target number of training samples.

    Returns:
        tuple: Two lists containing indices for the training and testing sets.
    """
    shuffled_indices = list(np.random.permutation(label_indices))
    train_indices = shuffled_indices[:train_target]
    test_indices = shuffled_indices[train_target:]
    return train_indices, test_indices

class LinearNet(nn.Module):
    """
    A simple linear neural network with one fully connected layer.
    The output is two nodes corresponding to two classes.

    Attributes:
    - fc1: Fully connected layer (input -> output)
    """

    def __init__(self, dimensions: int):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(dimensions, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels based on the highest probability from the output logits.
        """
        pred = self.forward(x)
        return torch.argmax(pred, dim=1)

class NonLinearNet(nn.Module):
    """
    A simple non-linear neural network with one hidden layer and Tanh activation.

    Attributes:
    - fc1: First fully connected layer (input -> hidden)
    - fc2: Second fully connected layer (hidden -> output)
    """

    def __init__(self, dimensions: int):
        super(NonLinearNet, self).__init__()
        self.fc1 = nn.Linear(dimensions, dimensions + 1)
        self.fc2 = nn.Linear(dimensions + 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels based on the highest probability from the output logits.
        """
        pred = self.forward(x)
        return torch.argmax(pred, dim=1)

def train_test_model(x_data, y_labels, input_dim, learning_rate, epochs, subsample_idx, model_info, model_type):
    """
    Trains and tests a model with a 2-fold subsampling procedure. Also performs adversarial attacks.

    Args:
        x_data (ndarray): The feature data.
        y_labels (ndarray): The class labels.
        input_dim (int): The dimensionality of the input data.
        learning_rate (float): The learning rate for the optimizer.
        epochs (int): The number of epochs for training.
        subsample_idx (int): The index of the subsample for the current iteration.
        model_info (str): A string indicating the model name or type.
        model_type (str): The type of model ('linear' or 'non-linear').

    Returns:
        dict: A dictionary containing training results and adversarial data.
    """
 
    # Set the flag for the case the attack fails
    repeat = True 
    while repeat==True:   
    
        # Split data into training and testing sets
        label0_indices = np.where(y_labels == 0)[0]
        label1_indices = np.where(y_labels == 1)[0]
        train_size = int(np.ceil(x_data.shape[0] / 2))
        
        # Calculate label ratio for subsampling
        label0_ratio = len(label0_indices) / (len(label0_indices) + len(label1_indices))
        label1_ratio = len(label1_indices) / (len(label0_indices) + len(label1_indices))
        label0_train_target = int(np.ceil(train_size * label0_ratio))
        label1_train_target = int(np.ceil(train_size * label1_ratio))
    
        # Create subsampling sets for both labels
        train0_indices, test0_indices = subsample_data(label0_indices, label0_train_target)
        train1_indices, test1_indices = subsample_data(label1_indices, label1_train_target)
        train_indices = np.concatenate([train0_indices, train1_indices])
        test_indices = np.concatenate([test0_indices, test1_indices])
        
        # Prepare training and testing data
        X_train = torch.tensor(x_data[train_indices], dtype=torch.float32)
        X_test = torch.tensor(x_data[test_indices], dtype=torch.float32)
        y_train = torch.tensor(y_labels[train_indices], dtype=torch.long)
        y_test = torch.tensor(y_labels[test_indices], dtype=torch.long)
    
        # Check for constant features (features with same value across all samples)
        nvalues = np.array([len(np.unique(X_train[:, dim])) for dim in range(input_dim)])
        keep_feature_indices = [dim for dim in range(input_dim) if nvalues[dim] != 1]
        
        X_train = X_train[:, keep_feature_indices]
        X_test = X_test[:, keep_feature_indices]
        num_features = len(keep_feature_indices)
    
        # Initialize the model
        if model_type == 'linear':
            model = LinearNet(num_features)
        elif model_type == 'non-linear':
            model = NonLinearNet(num_features)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
        # Set up loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
        # Train the model
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
    
        # Save the trained model
        model_name = f"{model_info}Model_Sub{str(subsample_idx)}.pt"
        torch.save(model.state_dict(), model_name)

        # Evaluate the model on the test set
        model.eval()
        y_pred_test = model.predict(X_test)
        test_accuracy = torch.mean((y_pred_test == y_test).float()).item()
    
        # Perform adversarial attack
        if model_type == 'linear':
            adv_data, adv_labels, repeat = fool.linear(X_train, y_train, model)
        elif model_type == 'non-linear':
            adv_data, adv_labels, repeat = fool.non_linear(X_train, y_train, model)

    # Package results
    results = {
        'train_indices': train_indices,
        'test_indices': test_indices,
        'test_predictions': y_pred_test.numpy(),
        'accuracy': test_accuracy,
        'adversarial_data': adv_data,
        'adversarial_labels': adv_labels,
        'useless_features': nvalues,
    }

    return results

