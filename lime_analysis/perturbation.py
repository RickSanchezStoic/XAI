from typing import Union, Dict, List

import torch
import numpy as np


def calculate_feature_means(
    dataset: Union[torch.Tensor, np.ndarray], feature_names: List[str]
) -> Dict[str, float]:
    """
    Calculates the mean of each feature across the entire dataset.

    Args:
        dataset (Union[torch.Tensor, np.ndarray]): The dataset containing the feature values (e.g., test dataset).
        feature_names (List[str]): List of feature names corresponding to the columns of the dataset.

    Returns:
        Dict[str, float]: A dictionary mapping feature names to their respective means.
    """
    if isinstance(dataset, np.ndarray):
        dataset = torch.tensor(dataset, dtype=torch.float32)

    feature_means = {
        feature: torch.mean(dataset[:, idx]).item()
        for idx, feature in enumerate(feature_names)
    }
    return feature_means


def perturb_single_instance(
    data_point: Dict[str, int],
    attributions: Union[torch.Tensor, np.ndarray],
    model: torch.nn.Module,
    dataset: Union[torch.Tensor, np.ndarray],
    feature_names: List[str],
) -> Dict[str, Union[List[int], int]]:
    """
    Perturbs a single instance and tracks when the prediction flips.

    Args:
        data_point (Dict[str, int]): Dictionary with label and index of the instance (e.g., {'label': index}).
        attributions (Union[torch.Tensor, np.ndarray]): Attributions generated by an explanation method.
        model (torch.nn.Module): The trained model used to make predictions.
        dataset (Union[torch.Tensor, np.ndarray]): The dataset (e.g., test set).
        feature_names (List[str]): List of feature names corresponding to the dataset.

    Returns:
        Dict[str, Union[List[int], int]]: Dictionary containing the steps at which the prediction flips (`flip_steps`),
                                          the original prediction (`original_prediction`), and the final perturbed prediction (`perturbed_prediction`).
    """
    label, index = list(data_point.items())[0]
    data_point_values = dataset[index].clone()

    if isinstance(attributions, np.ndarray):
        attributions = torch.tensor(attributions, dtype=torch.float32)

    feature_means = calculate_feature_means(dataset, feature_names)
    sorted_indices = torch.argsort(attributions, descending=True)

    # Get the original prediction
    original_prediction = model(data_point_values.unsqueeze(0)).argmax().item()

    flip_steps = []
    perturbed_data = data_point_values.clone()

    for i, feature_idx in enumerate(sorted_indices, start=1):
        feature_name = feature_names[feature_idx]
        perturbed_data[0][feature_idx] = feature_means[feature_name]

        # Get the perturbed prediction
        perturbed_prediction = model(perturbed_data.unsqueeze(0)).argmax().item()

        # If the prediction flips, record the step
        if perturbed_prediction != original_prediction:
            flip_steps.append(i)

    return {
        "flip_steps": flip_steps,
        "original_prediction": original_prediction,
        "perturbed_prediction": perturbed_prediction,
    }


def perturb_dataset(
    dataset: Union[torch.Tensor, np.ndarray],
    attributions: Union[torch.Tensor, np.ndarray],
    model: torch.nn.Module,
    feature_names: List[str],
) -> Dict[str, List[Union[int, float]]]:
    """
    Perturbs the entire dataset and computes accuracy after each perturbation step.

    Args:
        dataset (Union[torch.Tensor, np.ndarray]): The dataset to perturb (e.g., test set).
        attributions (Union[torch.Tensor, np.ndarray]): Attributions generated by an explanation method.
        model (torch.nn.Module): The trained model used for predictions.
        feature_names (List[str]): List of feature names corresponding to the dataset.

    Returns:
        Dict[str, List[Union[int, float]]]: A dictionary containing:
            - 'perturbation_steps': A list of integers representing the number of perturbation steps (number of features perturbed).
            - 'accuracies': A list of floats representing the accuracy at each perturbation step.
    """
    if isinstance(attributions, np.ndarray):
        attributions = torch.tensor(attributions, dtype=torch.float32)

    sorted_indices = torch.argsort(attributions, descending=True)
    feature_means = calculate_feature_means(dataset, feature_names)

    accuracy_results = {"perturbation_steps": [], "accuracies": []}
    original_predictions = model(dataset).argmax(dim=1)

    perturbed_dataset = dataset.clone()

    # Keep track of perturbed features to avoid redundant perturbations
    perturbed_features = set()

    for num_features in range(1, len(sorted_indices) + 1):
        # In each iteration, perturb the first 'num_features' features
        for i in range(num_features):
            feature_idx = sorted_indices[i]
            if feature_idx not in perturbed_features:
                feature_name = feature_names[feature_idx]
                perturbed_dataset[:, feature_idx] = feature_means[feature_name]
                perturbed_features.add(feature_idx)  # Mark the feature as perturbed

        perturbed_predictions = model(perturbed_dataset).argmax(dim=1)
        accuracy = (perturbed_predictions == original_predictions).float().mean().item()
        accuracy_results["perturbation_steps"].append(num_features)
        accuracy_results["accuracies"].append(accuracy)

    return accuracy_results


def compute_aupc(accuracy_results: Dict[str, List[float]]) -> float:
    """
    Computes the Area Under the Perturbation Curve (AUPC).

    Args:
        accuracy_results (Dict[str, List[float]]): A dictionary containing the perturbation steps and corresponding accuracies.
            - 'perturbation_steps': A list of integers representing the number of perturbation steps.
            - 'accuracies': A list of floats representing the accuracy at each perturbation step.

    Returns:
        float: The area under the perturbation curve, calculated using numerical integration (trapezoidal rule).
    """
    steps = accuracy_results["perturbation_steps"]
    accuracies = accuracy_results["accuracies"]
    aupc = np.trapz(
        accuracies, steps
    )  # Numerically integrate using the trapezoidal rule
    return aupc
