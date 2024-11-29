# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

"""
Runner Script for Adversarial Decoder
------------------------------------
This script runs the adversarial decoder on the dataset of choice.

Input:
    - Parameters file specifying configurations for the adversarial decoder.
    - Various helper functions for data loading, processing, and model training.

Output:
    - A Python dictionary holding adversarial decoding results, including:
        - Training/testing indices, predictions, and CCR values.
        - Identified global vectors/MIDs and their projections on the identified decision hyperplane.
        - Metrics such as "amount" and "consistency" describing MID dominance.
"""


import time
import pathlib
import yaml
from utils.timer import timer
from utils.data_loader import load_and_prepare_data
from main import main

# Record start time
start_time = time.time()

# Prompt user for parameter file
param_file_name = input('Enter the parameters file name (e.g., parameters.yml):\n')

# Load parameters from YAML file
with open(param_file_name, 'r') as file:
    parameters = yaml.safe_load(file)

# Extract and process parameters from the loaded YAML file
setcores     = int(parameters['core_id'])
mainpath     = parameters['main_path']
data_name    = parameters['dataset_name']
subsamplings = int(parameters['subsamplings'])
nshuffles    = int(parameters['shuffle_count'])
shuffletype  = parameters['shuffle_id']
model_flag   = parameters['model_id']
lr           = float(parameters['learning_rate'])
epochs       = int(parameters['epochs'])

# Parse ring domain
ring_start, ring_stop = int(5), int(35)

# Parse clustering parameters
min_samples, epsilon = int(3), float(0.25)

# Load data
x_data, y_labels, _, _ = load_and_prepare_data(mainpath, data_name)

# Prepare result directories
attack_info_path = f"{mainpath}/{data_name}_results/data_info/"
model_info_path = f"{mainpath}/{data_name}_results/model_info/"
pathlib.Path(model_info_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(attack_info_path).mkdir(parents=True, exist_ok=True)

# Validate data sufficiency
ones = sum(1 for x in y_labels if x == 1)
zeros = len(y_labels) - ones
if x_data.shape[1] <= 1 or ones <= 2 or zeros <= 2:
    raise ValueError('Not enough patterns per label or not enough dimensions. Try a different dataset.')

# Initialize results dictionary
att_res = {
    'parameters': {
        'model_attack_savepaths': [model_info_path, attack_info_path],
        'nsubsamplings': subsamplings,
        'model_flag': model_flag,
        'lr_epochs': [lr, epochs],
        'ring_domain': [ring_start, ring_stop],
        'min_samples_epsilon': [min_samples, epsilon]
    },
    'original_data': x_data,
    'original_labels': y_labels,
    'computation_cores': setcores
}

# Run adversarial classifier
main(att_res)

# Print elapsed time
timer(start_time)