import time

import numpy as np
import torch

def get_subset_samples(test_data, test_labels, model, device="cpu"):
    """
    Categorize samples into FP, TP, FN, TN and return their indices.

    Parameters:
    - test_data: torch.Tensor of test features.
    - test_labels: torch.Tensor of true test labels.
    - model: Trained PyTorch model.
    - device: Device to run the model on (default: 'cpu').

    Returns:
    - selected_indices: Dictionary containing indices for FP, TP, FN, TN.
    """

    # Ensure model is in evaluation mode and move data to the specified device
    model.eval()
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(test_data)
        predicted_labels = torch.argmax(outputs, dim=1)  # Convert logits to class predictions

    # Compute categories
    fp = (predicted_labels == 1) & (test_labels == 0)  # False Positive
    tp = (predicted_labels == 1) & (test_labels == 1)  # True Positive
    fn = (predicted_labels == 0) & (test_labels == 1)  # False Negative
    tn = (predicted_labels == 0) & (test_labels == 0)  # True Negative

    # Select one sample from each category (if available)
    selected_indices = {
        "FP": torch.where(fp)[0][:1].tolist(),
        "TP": torch.where(tp)[0][:1].tolist(),
        "FN": torch.where(fn)[0][:1].tolist(),
        "TN": torch.where(tn)[0][:1].tolist(),
    }

    return selected_indices




def measure_time(func):
    """
    Decorator to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start the timer
        result = func(*args, **kwargs)  # Execute the function
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        return result, elapsed_time  # Return result and elapsed time
    return wrapper





def absolute_max_scaling(scores):
    """
    Normalizes the attribution scores using absolute max scaling.

    Parameters:
        scores (array-like): Attribution scores to be normalized.

    Returns:
        np.ndarray: Normalized scores in the range [-1, 1].
    """
    scores = np.array(scores)  # Ensure input is a NumPy array
    max_abs_value = np.max(np.abs(scores))
    if max_abs_value == 0:
        return scores  # Return original scores if all values are zero
    return scores / max_abs_value

