# Automated Driving Environment Models

This repository contains a Python and Jupyter Notebook based project for modelling environment representations used in automated driving. The main focus of this project is **occupancy grid mapping**, which is an important technique for representing free space, occupied space, and unknown areas around an autonomous vehicle.

The repository includes notebooks, utility files, and lab instructions for working with automated driving environment models. The project is mainly written in Jupyter Notebook with some Python utility code. The repository structure currently includes `env_utils`, `representations`, `occupancy_grid.ipynb`, `environment.yml`, and `lab_instructions.pdf`. 

## Overview

Autonomous vehicles need to understand their surrounding environment before making driving decisions. One common way to represent the environment is by using an **occupancy grid**.

In an occupancy grid, the environment is divided into small grid cells. Each cell stores information about whether that area is:

- Free
- Occupied
- Unknown

This type of representation is useful for perception, mapping, localization, path planning, and decision-making in automated driving systems.

## Project Goal

The goal of this project is to understand how environment data can be processed and converted into useful driving environment representations.

The main objectives are:

- Understand occupancy grid representation
- Process automated driving environment data
- Visualize raw data
- Generate grid-based environment models
- Use Python and Jupyter Notebook for experimentation

## Features

- Occupancy grid modelling
- Raw data visualization
- Jupyter Notebook based workflow
- Python utility functions
- Conda environment setup
- Educational lab instructions

## Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Matplotlib
- Conda
- Lanelet2
- PyProj

## Project Structure

```text
automated-driving-environment-models/
│
├── env_utils/
│   ├── setup.py
│   └── src/
│       └── Utility source files for environment modelling
│
├── representations/
│   └── raw_data_visualization.ipynb
│       └── Notebook for visualizing raw environment data
│
├── .gitignore
│   └── Git ignore file
│
├── Untitled.ipynb
│   └── Additional experimental notebook
│
├── environment.yml
│   └── Conda environment configuration file
│
├── lab_instructions.pdf
│   └── Lab instructions and project description
│
└── occupancy_grid.ipynb
    └── Main notebook for occupancy grid modelling
```

## Main Files

### `occupancy_grid.ipynb`

This is the main notebook of the project. It contains the implementation and explanation of occupancy grid modelling for automated driving environments.

### `representations/raw_data_visualization.ipynb`

This notebook is used to visualize raw environment data before converting it into another representation.

### `env_utils/`

This folder contains helper utilities and setup files used in the project.

### `environment.yml`

This file contains the Conda environment configuration. It helps to install the required dependencies.

### `lab_instructions.pdf`

This file contains the lab instructions or project description.

## Installation

First, clone the repository:

```bash
git clone https://github.com/PrithviRaj037/automated-driving-environment-models.git
```

Go into the project folder:

```bash
cd automated-driving-environment-models
```

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate teaching
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open the main notebook:

```text
occupancy_grid.ipynb
```

## Usage

1. Clone the repository.
2. Create and activate the Conda environment.
3. Open Jupyter Notebook.
4. Run `occupancy_grid.ipynb`.
5. Follow the notebook cells step by step.
6. Visualize and analyze the occupancy grid output.

## Occupancy Grid Concept

An occupancy grid is a grid-based map representation. The environment is divided into small cells, and each cell describes the state of that area.

A typical occupancy grid can represent:

```text
Free space      → Area where the vehicle can move
Occupied space  → Area blocked by obstacles
Unknown space   → Area not yet observed
```

This representation is useful because it converts complex real-world driving scenes into a structured format that can be used by autonomous driving algorithms.

## Example Workflow

```text
Raw driving environment data
            ↓
Data preprocessing
            ↓
Coordinate transformation
            ↓
Occupancy grid generation
            ↓
Visualization
            ↓
Environment representation for automated driving
```

## Applications

This project is related to:

- Automated driving
- Autonomous vehicles
- Mobile robotics
- Environment perception
- Occupancy grid mapping
- Path planning
- Scene understanding

## Future Improvements

Possible future improvements include:

- Add example output images
- Add more detailed notebook explanations
- Add more environment representations
- Improve code modularity
- Add testing for utility functions
- Add simulation examples
- Add comparison between different map representations

## Author

**Prithvi Raj**

GitHub: [PrithviRaj037](https://github.com/PrithviRaj037)

## License

This project is currently used for educational and research purposes.  
