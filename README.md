# QICS: Quantum-Inspired Classical Systems Analysis

This repository contains Python code for the article on quantum-inspired classical systems analysis. It includes scripts for generating and analyzing data using a modified Lorenz96 model, real-world temperature data, and white noise.

## Project Structure

- **Lorenz96/**: Scripts for the modified Lorenz96 model.
  - `generate.py`: Generates data for the modified Lorenz96 model.
  - `partialed.py`: Performs analysis on the generated Lorenz96 data.
- **Temperature analysis/**: Analyzes real temperature data.
  - `data/`: Contains small CSV files with temperature data.
  - `Temperature_analysis.py`: Analyzes data from the `data/` folder.
- **White Noise/**: Generates and analyzes white noise.
  - `White-noise_analysis.py`: Creates white noise data and performs analysis.

## Prerequisites

- Python 3.x
- Required libraries listed in `requirements.txt`:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `qutip`
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

1. **Lorenz96 Analysis**:
   - Run `generate.py` to create Lorenz96 model data:
     ```bash
     python Lorenz96/generate.py
     ```
   - Then run `partialed.py` to analyze the data:
     ```bash
     python Lorenz96/partialed.py
     ```
   - Adjust parameters in both scripts as needed (see code comments for details).

2. **Temperature Analysis**:
   - Ensure CSV files are in `Temperature analysis/data/`.
   - Run the analysis script:
     ```bash
     python "Temperature analysis/Temperature_analysis.py"
     ```
   - Modify parameters in `Temperature_analysis.py` as desired.

3. **White Noise Analysis**:
   - Run the white noise script to generate and analyze data:
     ```bash
     python "White Noise/White-noise_analysis.py"
     ```
   - Update parameters in `White-noise_analysis.py` to customize.

## Notes

- Each script includes configurable parameters (check code comments for details).
- The CSV files in `Temperature analysis/data/` are small and included in the repository.
- For questions or issues, refer to the article or open an issue on this repository.