# Most Informative Directions (MIDs) for binary classification tasks (Adversarial Decoder Project)

## Overview
The Adversarial Decoder is a Python-based project designed to visualize classification decision hyperplanes, providing insight on the features with high information content. This is represented by the MID vectors, which possess a weight value for each of the input data features. To do so the Adversarial Decoder takes advantage of gradient-based adversarial attacks in order to probe the decision hyperplane in a computationally efficient manner. Additionally, the identified MIDs are subsequently used as a denoiser, sorting out the data points high in information content, which are more influential for the shape of the decision boundary and are not viewed as noise by the classification algorithm. We termed these data points relevant patterns (RPs). The project provides a pipeline for subsampling, clustering, and inference to extract meaningful insights from datasets modified by generative adversarial attacks. The project uses a modular architecture, integrating several helper functions to streamline the decoding process.

## Features
- Parallelized subsampling using multiple CPU cores.
- Modular design with distinct helper functions for tasks such as clustering, eigendecomposition, and projection computation and analysis.
- Support for both linear and more importantly non-linear models.
- Easily configurable parameters for adapting to various datasets and attack scenarios.

## Installation

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/username/adversarial-decoder.git
   cd adversarial-decoder
   ```

2. Install dependencies:
   
   You can choose between using `pip` or `conda` for dependency management:

   #### Using `pip`:
   If you prefer `pip`, use the `requirements.txt` file:
   ```bash
   pip install -r requirements_ma.txt
   ```

   #### Using `conda`:
   If you prefer `conda`, use the provided environment file to create a new environment:
   ```bash
   conda env create -f env_ma.yml

---

## Project Structure
```
adversarial-decoder/
├── run_MIDs.py          # Main script for the adversarial decoder.
├── main.py              # Core computation logic
├── utils/               # Contains helper functions.
│    ├── data_loader.py            
│    ├── test_train_model.py            
│    ├── attack_foolbox.py            
│    ├── clustering.py            
│    ├── eigendecomposition.py    
│    ├── inference.py             
│    ├── neighborhood_computation.py          
│    ├── projections.py
│    ├── gravity_center.py
│    ├── timer.py
│    ├── relevance.py                                                                      
│    └── unification.py           
├── results/                # Directory to store results.
├── requirements_ma.txt     # List of dependencies.
├── env_ma.yml              # Conda environment with dependencies.
├── README.md               # Project documentation.
└── LICENSE                 # License information.
```

## Usage

1. **Prepare Input Parameters**:
   Create a parameter file (see `parameters.yml`) with the suggested format:

Note: Ensure your dataset (e.g., hello_world.pkl) is a valid Python pickle file containing a dictionary with keys **data** and **labels** that match the expected format (see hello_world.pkl).

2. **Run the Adversarial Decoder**:
   Execute the test using the `run_MIDs.py` script:
   ```bash
   python significance/run_MIDs.py
   ```
   You will be prompted to enter the parameter file name (parameters.yml).

3. **Results**:
   The results will be saved in the specified `main path` under a `data_info/` directory. Results include:
   - Pickle file with CCR metrics.
   - Pickle file with identified MIDs and their respective metrics.
   - Pickle file with the identified RPs and their respective metrics. 

---

## Contributing

We welcome contributions to improve this framework. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the [MIT License](LICENSE).

