# abm-emulator
This directory contains code and data for building Gaussian process emulators of agent-based model outputs
using Kronecker-structured Gaussian processes in PyMC3. Jupyter notebooks and Python libraries are included in the `python/` directory, while the datasets are stored in `data/`. Note that processed versions of each dataset can be found in the `data/processed-abm-inputs` directory. These are formatted in JSON files and are structured as dictionaries with key-value pairs for all the information required to build an emulator, including the coordinates for each observation as well as the response variable associated with each coordinate. For long-running processes, log files are stored in the `logs/` directory. Directories labeled `old` include data and outputs from previous iterations that are no longer used.

# Visualizations
Images and figures are stored in the `figures/` directory. Documents are stored in the `notes/` directory.

# Dependencies
To ensure all dependencies are available, use the `requirements.txt` file located in the top directory to get all the necessary packages. You may run the command `pip install -r requirements.txt` from the top directory to install then.
