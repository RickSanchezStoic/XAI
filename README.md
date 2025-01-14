# README: Explainable AI Library for Binary Classification

## Overview
This library helps explain how machine learning models make predictions, focusing on binary classification tasks. It uses simple and flexible parts that work together, making it easy to expand in the future with more explanation methods.The main feature is LIME (Local Interpretable Model-agnostic Explanations), which helps users understand predictions in detail. The project is well-organized, with each part handling a specific task. The main.ipynb file shows the complete process, from preparing data to training models and creating explanations, making it easy to follow and use


---

## Project Structure
```
|-- data
    |-- breast_data.py
    |-- __init__.py
|-- lime_analysis
    |-- lime_explanations.py
    |-- perturbation.py
    |-- utils.py
    |-- __init__.py
|-- main.ipynb
|-- model
    |-- linear.py
    |-- loss.py
    |-- training.py
    |-- __init__.py
|-- test.py
|-- visualisation
    |-- lime.py
    |-- metrics.py
    |-- __init__.py
```

### Key Components
- **`data/`**: Code for dataset preparation and loading.
  - `breast_data.py`: Prepares the Breast Cancer Wisconsin dataset.

- **`lime_analysis/`**: Handles LIME-based explanations and related evaluation.
  - `lime_explanations.py`: Generates LIME explanations.
  - `perturbation.py`: Implements perturbation analysis for explanation evaluation.
  - `utils.py`: Provides utility functions for explanation and evaluation.

- **`main.ipynb`**: Jupyter Notebook with the complete pipeline for model training, evaluation, and explanation generation.

- **`model/`**: Contains the model architecture, loss functions, and training pipeline.
  - `linear.py`: Defines a linear neural network for binary classification.
  - `loss.py`: Custom loss functions (if required).
  - `training.py`: Manages training and evaluation logic.

- **`visualisation/`**: Tools for visualizing model performance and explanations.
  - `lime.py`: Visualizes LIME-generated explanations.
  - `metrics.py`: Plots performance metrics such as accuracy, F1-score, and ROC curves.

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- Main Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `torch`
  - `lime`
  - `captum`
  - `matplotlib`
  - `seaborn`

### Installation Steps
1. Clone the repository:
   ```bash
   git clone [repository URL]
   cd [repository folder]
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Linux/macOS Setup (using Conda):

    Execute the setup.sh script to install dependencies and set up the environment:
   
    ```
    chmod +x setup.sh  # Make the script executable (if not already)
    ./setup.sh
    ```
4. Windows Setup (using Conda):

    Execute the setup.bat script to install dependencies and set up the environment:
    ```
    setup.bat
    ```
5.  Install additional dependencies (if needed):
    ```
    pip install -r requirements.txt
    ```

---

## Execution Guide

### 1. Build and Train the Model
- Use the linear neural network defined in `model/linear.py`.
- Train and evaluate the model using `model/training.py`.
- Evaluate key metrics, including accuracy, F1-score, and ROC-AUC.

### 2. Generate LIME Explanations
- Select test samples and generate explanations with `lime_analysis/lime_explanations.py`.
- Evaluate explanations using perturbation analysis (`lime_analysis/perturbation.py`).

### 3. Visualization and Analysis
- Visualize performance metrics with `visualisation/metrics.py`.
- Display LIME explanations using `visualisation/lime.py`.

### 4. Run the Complete Workflow
- The `main.ipynb` notebook demonstrates the end-to-end process, integrating all the steps mentioned above. It includes detailed code and explanations for every task, making it easy to replicate and extend.

---

## Output

### Key Deliverables
1. **Trained Model**: A binary classifier trained on the Breast Cancer Wisconsin dataset.
2. **Explanations**: LIME-based insights for model predictions.
3. **Performance Metrics**: Accuracy, F1-score, ROC-AUC, and visual plots.
4. **Notebook**: Complete implementation in `main.ipynb` for easy replication.

---

## Contact
For queries or clarifications, please contact:
- Name: Rishabh Kumar
- Email: rishabh.kumar@edu.rptu.de



