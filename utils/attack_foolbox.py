# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

import numpy as np
import foolbox
import torch

"""
perform_attack
--------------------
This function performs adversarial attacks on the given model. 
It supports both linear (FGSM) and non-linear (PGD) models.

Arguments:
    X_train (torch.Tensor): Input features.
    y_train (torch.Tensor): Labels corresponding to the input features.
    model (torch.nn.Module): The trained model to attack.
    attack_type (str): Specifies the type of attack ('linear' or 'non_linear').

Returns:
    tuple: Adversarial data and labels from both attacks. 
           If an attack fails, it returns zeros and a repeat flag.
"""

def linear(X_train,y_train,model):
    
    # Define the bounds for the attack based on input data
    bound_min = np.min(X_train.numpy())
    bound_max = np.max(X_train.numpy())
    
    # Set the trained model for the attack
    fmodel = foolbox.models.PyTorchModel(model.eval(), bounds=(bound_min,bound_max), num_classes=2)
    # Choose attack
    attack = foolbox.attacks.FGSM(fmodel)
    
    # Set Data Format
    images = X_train.numpy()
    original_labels = y_train.numpy()
    # Get Adversarial Data
    adversarial_data = attack(images, original_labels,epsilons=100) 


    if np.any(np.isnan(adversarial_data)) or np.any(np.isinf(adversarial_data)):
        repeat = True
        #print('....Attack 1 Failed')
        return 0,0,repeat
    
    # Get Adversarial Labels
    adversarial_labels = fmodel.forward(adversarial_data).argmax(axis=-1)

    # Set the trained model for the attack
    model_adv = model    
    fmodel = foolbox.models.PyTorchModel(model_adv.eval(), bounds=(bound_min,bound_max), num_classes=2)
    
    # Bring data to proper form for the second attack
    adversarial_data, adversarial_labels = map(torch.tensor, (adversarial_data,adversarial_labels))
    adversarial_data = adversarial_data.float()
    adversarial_labels = adversarial_labels.long()
    images = adversarial_data.numpy()
    adversarial_labels = adversarial_labels.numpy()        
    # Choose attack
    attack = foolbox.attacks.FGSM(fmodel)
    
    # Get Adversarial Data
    adversarial_data = attack(images, adversarial_labels,epsilons=100)

    if np.any(np.isnan(adversarial_data)) or np.any(np.isinf(adversarial_data)):
        repeat = True
        #print('....Attack 2 Failed')
        return 0,0,repeat

    # Get Adversarial Labels
    adversarial_labels = fmodel.forward(adversarial_data).argmax(axis=-1)
    repeat = False
    
    return adversarial_data, adversarial_labels, repeat



def non_linear(X_train,y_train,model):

    # Define the bounds for the attack based on input data
    bound_min = np.min(X_train.numpy())
    bound_max = np.max(X_train.numpy())
    
    # Set the trained model for the attack
    fmodel = foolbox.models.PyTorchModel(model.eval(), bounds=(bound_min,bound_max), num_classes=2)
    # Choose attack
    attack = foolbox.attacks.PGD(fmodel,distance=foolbox.distances.Linf)
    
    # Set Data Format
    images = X_train.numpy()
    original_labels = y_train.numpy()
    # Get Adversarial Data
    adversarial_data = attack(images, original_labels) 

    if np.any(np.isnan(adversarial_data)) or np.any(np.isinf(adversarial_data)):
        repeat = True
        #print('....Attack 1 Failed')
        return 0,0,repeat
    
    # Get Adversarial Labels
    adversarial_labels = fmodel.forward(adversarial_data).argmax(axis=-1)

    # Set the trained model for the attack
    model_adv = model    
    fmodel = foolbox.models.PyTorchModel(model_adv.eval(), bounds=(bound_min,bound_max), num_classes=2)
    
    # Bring data to proper form for the second attack
    adversarial_data, adversarial_labels = map(torch.tensor, (adversarial_data,adversarial_labels))
    adversarial_data = adversarial_data.float()
    adversarial_labels = adversarial_labels.long()
    images = adversarial_data.numpy()
    adversarial_labels = adversarial_labels.numpy()        
    # Choose attack
    attack = foolbox.attacks.PGD(fmodel,distance=foolbox.distances.Linf)
    
    # Get Adversarial Data
    adversarial_data = attack(images, adversarial_labels)

    if np.any(np.isnan(adversarial_data)) or np.any(np.isinf(adversarial_data)):
        repeat = True
        #print('....Attack 2 Failed')
        return 0,0,repeat
    
    # Get Adversarial Labels
    adversarial_labels = fmodel.forward(adversarial_data).argmax(axis=-1)
    repeat = False
    
    return adversarial_data, adversarial_labels, repeat



