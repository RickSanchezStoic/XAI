import random
import time
from typing import Dict, List, Callable, Tuple, Any, Union

import numpy as np
import torch


def get_subset_samples(
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    model: torch.nn.Module,
    device: str = "cpu",
    use_random: bool = False,
) -> Dict[str, List[int]]:
    """
    Categorize samples into FP, TP, FN, TN and return their indices.
    When `use_random` is set to True, random data points are selected for each category.

    Args:
        test_data (torch.Tensor): Tensor of test features (e.g., input data for classification).
        test_labels (torch.Tensor): Tensor of true test labels (e.g., ground truth labels for classification).
        model (torch.nn.Module): The trained PyTorch model used for prediction.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        use_random (bool, optional): If True, selects random samples. If False, selects the first available sample. Defaults to False.

    Returns:
        Dict[str, List[int]]: A dictionary containing lists of indices for False Positive (FP), True Positive (TP),
                               False Negative (FN), and True Negative (TN).
                               - 'FP': List of indices where predicted labels are 1 and true labels are 0.
                               - 'TP': List of indices where predicted labels are 1 and true labels are 1.
                               - 'FN': List of indices where predicted labels are 0 and true labels are 1.
                               - 'TN': List of indices where predicted labels are 0 and true labels are 0.
    """
    # Ensure model is in evaluation mode and move data to the specified device
    model.eval()
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(test_data)
        predicted_labels = torch.argmax(
            outputs, dim=1
        )  # Convert logits to class predictions

    # Compute categories (FP, TP, FN, TN)
    fp = (predicted_labels == 1) & (test_labels == 0)  # False Positive
    tp = (predicted_labels == 1) & (test_labels == 1)  # True Positive
    fn = (predicted_labels == 0) & (test_labels == 1)  # False Negative
    tn = (predicted_labels == 0) & (test_labels == 0)  # True Negative

    # Select random sample from each category (if available)
    selected_indices = {}
    categories = {"FP": fp, "TP": tp, "FN": fn, "TN": tn}

    for category, mask in categories.items():
        indices = torch.where(mask)[
            0
        ].tolist()  # Get all indices that match the condition

        if indices:
            if use_random:
                random.seed()
                # Randomly select an index
                selected_indices[category] = [random.choice(indices)]
            else:
                # Choose the first index
                selected_indices[category] = [indices[0]]
        else:
            # If no samples in the category, set it to an empty list
            selected_indices[category] = []

    return selected_indices


def measure_time(func: Callable[..., Any]) -> Callable[..., Tuple[Any, float]]:
    """
    Decorator to measure the execution time of a function.

    Args:
        func (Callable[..., Any]): The function to decorate. It can take any number of positional and keyword arguments
                                   and return any type of result.

    Returns:
        Callable[..., Tuple[Any, float]]: A decorated function that returns a tuple containing the original function's result
                                          and the execution time in seconds.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
        start_time = time.time()  # Start the timer
        result = func(*args, **kwargs)  # Execute the function
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        return result, elapsed_time  # Return result and elapsed time

    return wrapper


def absolute_max_scaling(scores: Union[np.ndarray, list]) -> np.ndarray:
    """
    Normalizes the attribution scores using absolute max scaling.

    This method scales the input attribution scores so that the largest absolute
    value of the scores is mapped to 1, and all other scores are scaled proportionally
    between -1 and 1.

    Args:
        scores (Union[np.ndarray, list]): Attribution scores to be normalized,
                                           which can be a NumPy array or a list.

    Returns:
        np.ndarray: Normalized scores in the range [-1, 1]. If all input scores are zero,
                    the original scores are returned without scaling.
    """
    scores = np.array(scores)  # Ensure input is a NumPy array
    max_abs_value = np.max(
        np.abs(scores)
    )  # Find the maximum absolute value in the scores
    if max_abs_value == 0:
        return scores  # Return original scores if all values are zero
    return (
        scores / max_abs_value
    )  # Normalize the scores by dividing by the maximum absolute value
