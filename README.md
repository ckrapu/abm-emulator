# abm-emulator
This directory contains code and data for building Gaussian process emulators of agent-based model outputs
using Kronecker-structured Gaussian processes in PyMC3. Jupyter notebooks and Python libraries are included in the `python/` directory, while the datasets are stored in `data/`. Note that processed versions of each dataset can be found in the `data/processed-abm-inputs` directory. These are formatted in JSON files and are structured as dictionaries with key-value pairs for all the information required to build an emulator, including the coordinates for each observation as well as the response variable associated with each coordinate. For long-running processes, log files are stored in the `logs/` directory. Directories labeled `old` include data and outputs from previous iterations that are no longer used.

# Visualizations
Images and figures are stored in the `figures/` directory. Documents are stored in the `notes/` directory.

# Dependencies
To ensure all dependencies are available, use the `requirements.txt` file located in the top directory to get all the necessary packages. You may run the command `pip install -r requirements.txt` from the top directory to install then.

# Bash script
This section explains the usage of the Bash script located at `run-experiments.sh`, which is used to execute experiments for fitting Gaussian Process (GP) emulators.

### Loop Execution
- The script contains nested loops to iterate over the datasets, model types, and fitting methods.
- Within the nested loops, the script sets the log and output paths based on the current dataset, model type, and fitting method.
- It removes any existing log file corresponding to the current combination of parameters.
- It then executes a Python command using the `nohup` command to run the `src/cli.py` script with the `fit_model` command, passing the appropriate input arguments.
- The output of the command is redirected to the log file.
- This loop structure ensures that the fitting process is executed for all combinations of datasets, model types, and fitting methods.

## Conclusion
The provided Bash script automates the execution of experiments for fitting GP emulators. It loops over different datasets, model types, and fitting methods, and executes the `fit_model` command using the `src/cli.py` script. This script facilitates the efficient execution of multiple experiments with different configurations.

# Emulator models
This section explains the Python code located at `src/emulators.py`, which provides classes and functions for fitting and predicting with Gaussian Process (GP) emulators.

## Code Overview
The provided Python code consists of several classes and functions:

### Class: PyMC3WrapperGP
- This class serves as a base class for the emulators implemented using PyMC3.
- It provides common functionality and initialization parameters for the emulators.
- The `fit` method fits the GP model using Markov Chain Monte Carlo (MCMC) or Variational Inference (VI) methods.
- The `predict` method generates predictions at new spatiotemporal coordinates using the fitted GP model.

### Class: Independent
- This class represents an independent GP emulator.
- It inherits from the `PyMC3WrapperGP` class.
- The `fit` method fits separate GPs for each coordinate array and response values.
- The `predict` method generates predictions at new coordinates using the separate GPs.

### Class: SpaceTimeKron
- This class represents a space-time Kronecker GP emulator.
- It inherits from the `PyMC3WrapperGP` class.
- The `fit` method fits a Kronecker GP model using the spatiotemporal coordinates and response values.
- The `predict` method generates predictions at new spatiotemporal coordinates using the fitted Kronecker GP model.

### Other Classes and Functions
- `SpaceTimeKronGPy`: Class for fitting and predicting with a Kronecker Matern GP using GPy.
- `IndependentGR`: Class for fitting and predicting with independent GPs using GPy.
- `SpaceTimeIndependentGPy`: Class for fitting and predicting with independent GPs for each coordinate using GPy.
- `length_bbox_diagonal`: Function to calculate the Euclidean distance between diagonally opposite corners of a bounding hypervolume.
- `autocalc_lengthscale_bounds`: Function to automatically determine bounds on the lengthscales based on input data.
- `make_kernel`: Function to create a Matern52 kernel for GPy.
- `vary_ndims`: Function to create new coordinates by varying certain dimensions while keeping others constant.


# SIR model
This section explains the Python code located at `src/abmlib.py`, which implements an agent-based model for infection spread. The model simulates the spread of an infection among agents in a two-dimensional grid.

## Code Overview
The provided Python code consists of several classes and functions:

### Class: InfectionModel
- This class represents the overall infection model.
- It initializes the model with parameters such as the number of agents, grid dimensions, infection probabilities, recovery rates, and death rates.
- The `__init__` method creates agents, assigns them to random positions on the grid, and infects some agents at the start.
- The `get_recovery_time` method returns a randomly sampled recovery time for an infected agent.
- The `step` method advances the model by one time step, collecting data using a `DataCollector` and updating the schedule of agents.

### Enum: State
- This enumeration defines the possible states for agents: SUSCEPTIBLE, INFECTED, and REMOVED.

### Class: MyAgent
- This class represents an agent in the infection model.
- Each agent can be in one of the defined states and has attributes such as infection time, doctor status, and position on the grid.
- The `move` method moves the agent to a neighboring grid cell based on Moore's neighborhood.
- The `status` method checks the infection status of the agent and updates its state accordingly.
- The `contact` method identifies close contacts with other agents and infects susceptible agents or cures infected agents based on probabilities.
- The `step` method executes the agent's actions in a single time step.

### Function: SIR
- This function simulates the infection model for a specified number of steps.
- It initializes the model, iterates over the steps, and updates the grid state and doctor locations at each step.
- The function returns the final model object, the grid state at each step, and the doctor locations at each step.

### Other Utilities
- The `state_dict` dictionary maps state integers to state labels for convenient visualization.
