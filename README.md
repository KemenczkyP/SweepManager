# SweepManager

## Overview
The Sweep Manager is a Python-based tool designed to simplify and optimize the process of 
hyperparameter tuning in machine learning experiments. Utilizing Optuna for optimization 
and MLFlow for experiment tracking, this utility provides an integrated approach to 
managing and logging the performance of various hyperparameter configurations. This tool 
is designed with flexibility in mind, supporting a wide range of machine learning tasks and models.

## Features
 - Hyperparameter Optimization: Leverages Optuna to systematically explore the hyperparameter space and identify the optimal configuration for your model.
 - Experiment Tracking: Integrates with MLFlow to log experiments, track progress, and store results for future analysis.
 - Scalability: Designed to accommodate future expansion with more ML Ops-related functionalities and tools.
 - Flexibility: Supports various hyperparameter types, including categorical, integer, and float, allowing for comprehensive experimentation.

## Installation
To use the Sweep Manager, you need Python 3.8 or later. 
### Install as a pip package
```bash
pip install git+https://github.com/KemenczkyP/SweepManager
```
The primary dependencies are optuna and mlflow, which will be installed alongside other necessary packages.

## Usage
 - Configure MLFlow: Before running optimizations, configure MLFlow with your experiment name and tracking URI using the configure_MLFlow method of the SweepManager class.
 - Define Your Objective Function: Create a function that takes hyperparameters as input and returns a performance metric (e.g., accuracy, loss) as output. This function should incorporate the model training and evaluation logic.
 - Optimize: Use the run_optuna_with_mlflow method to start the optimization process. This method takes your objective function as an argument and utilizes Optuna to optimize the hyperparameters based on the specified direction (minimize or maximize).

```python
from sweep_manager.sweep_manager import SweepManager

# Define your train function
def train_model(hyperparameters, run_name):
    # Your model training logic here
    # Return the performance metric (e.g., accuracy, loss)
    metrics: dict = {}
    return metrics

# Initialize SweepManager
sweep_manager = SweepManager(sweep_id="example_sweep", num_trials=100, optuna_direction="minimize")

# Configure MLFlow
tracking_uri = f"file:/absolute/path/to/output/folder" + "/runs/mlflow"  # OR URI e.g. "http://localhost:5000"
sweep_manager.configure_MLFlow(experiment_name="MyExperiment", tracking_uri=tracking_uri)

# Get objective function with hyperparameters and train function
objective = sweep_manager.get_objective("path/to/hyperparameters.json", train_function=train_model)

# Run optimization with MLFlow integration
sweep_manager.run_optuna_with_mlflow(objective)

```

 - Start MLFlow server with the following command (on local or remote machine):
```bash
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri file:/absolute/path/to/output/folder/runs/mlflow --default-artifact-root artifacts 
```
 
 - Run the script and monitor the progress in the MLFlow UI.


## Hyperparameter file
It uses the `optuna` library for hyperparameter optimization. 
The hyperparameters are defined in the `hyperparameters.json` file.

 - `hyperparameters.json` should contain the hyperparameters for the training process.
The json file contains two fields: `help` and `hyperparameters`. In the help, the description of
each hyperparameter is given.
The hyperparameters file contains two types of hyperparameters: `iter` and `range`. 
   - If `iter` is present, the 
hyperparameter optimizer will search in the list. 
   - If `range` is present, the hyperparameter is numerical and the optimizer will 
   search in the range.
   - If both are present, the optimizer will use the `iter` values and ignore the `range`.
   - If none is present, the optimizer will use the default value. If the default value is not present, the script will throw an error.
     ```json
          {
          "optimizer": {
            "type": "categorical",
            "default": "Adam",
            "iter": [
              "SGD",
              "Adam",
              "AdamW",
              "NAdam",
              "RAdam",
              "RMSProp"
              ]
          },
          "cls": {
            "type": "float",
            "default": 0.5,
            "range": [
              0.1,
              0.9
            ]
          }
        }
     ```

