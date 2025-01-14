# README: Binary Classification with LIME Explanations

## Overview
This project involves building, training, and evaluating a binary classification model on the Breast Cancer Wisconsin dataset. The task also includes generating and analyzing explanations for the model's predictions using LIME and other XAI techniques. The following sections outline the key steps, tools, and files included in this project.

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

### Key Directories and Files
- **`data/`**: Contains the code for dataset preparation and loading.
  - `breast_data.py`: Prepares the Breast Cancer Wisconsin dataset.

- **`lime_analysis/`**: Handles LIME explanations and evaluation.
  - `lime_explanations.py`: Generates explanations using LIME.
  - `perturbation.py`: Implements perturbation analysis.
  - `utils.py`: Utility functions for explanation and evaluation.

- **`main.ipynb`**: Jupyter Notebook containing the end-to-end implementation of the task.

- **`model/`**: Contains the model architecture, loss functions, and training logic.
  - `linear.py`: Defines the linear neural network.
  - `loss.py`: Custom loss functions (if any).
  - `training.py`: Handles model training and evaluation.

- **`visualisation/`**: Visualization of model performance and explanations.
  - `lime.py`: Visualizes LIME explanations.
  - `metrics.py`: Plots performance metrics (e.g., accuracy, F1-score, ROC).

---

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `torch`
  - `lime`
  - `captum`
  - `matplotlib`
  - `seaborn`

### Installation
1. Clone the repository:
   ```bash
   git clone [repository URL]
   cd [repository folder]
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation
1. Download the Breast Cancer Wisconsin dataset from:
   - [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
2. Save the dataset in a suitable location (if required).
3. Load the dataset using `data/breast_data.py`.

---

## Steps to Execute

### 1. Build and Train the Model
- **Model Architecture**: Linear neural network implemented in `model/linear.py`.
- **Training**: Train the model using the logic in `model/training.py`.
- **Evaluation Metrics**:
  - Accuracy over epochs
  - F1-Score over epochs
  - ROC Curve and AUC

### 2. Generate Explanations with LIME
- Select a subset of test samples.
- Generate explanations using `lime_analysis/lime_explanations.py`.

### 3. Evaluate Explanations
- Metrics:
  - **Time**: Time taken to generate explanations.
  - **Parsimony**: Count of features with importance above a threshold.
  - **Correctness**: Perturbation analysis using `lime_analysis/perturbation.py`.

### 4. Visualization
- Performance metrics plotted using `visualisation/metrics.py`.
- LIME explanations visualized in `visualisation/lime.py`.

---

## Output

### Files Produced
1. **Jupyter Notebook** (`main.ipynb`): Contains all code and explanations.
2. **Rendered Notebook**: HTML/PDF output of `main.ipynb` with results.
3. **Supporting Scripts**:
   - Python scripts used as imports in the notebook.

---

## Evaluation Metrics
- **Model Performance**:
  - Accuracy, F1-Score, ROC-AUC
- **Explanation Evaluation**:
  - Average time, parsimony, and correctness (perturbation analysis).

---

## Challenges and Notes
### Challenges Encountered
_TODO: Describe any significant challenges or limitations faced during the implementation._

### Assumptions and Simplifications
_TODO: Specify any assumptions or simplifications made._

---

## Appendix
- **Perturbation Curve**: Explanation and methodology used.
- **Scaling Technique**: Details on feature scaling for parsimony analysis.

---

## Contact
For queries or clarifications, please contact:
- Name: [Your Name]
- Email: [Your Email]

---

## License
_TODO: Specify the license under which this project is shared._

