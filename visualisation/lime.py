from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_perturbation_steps(perturbation_results: Dict[str, list]) -> None:
    """
    Plots the perturbation steps and indicates when the prediction flips.

    Args:
        perturbation_results (dict): Dictionary containing 'flip_steps', and 'original_prediction'.
            - 'flip_steps' (list): List of perturbation steps where the prediction flipped.
            - 'original_prediction' (int): The original model's prediction before perturbation.

    Returns:
        None: Displays a plot showing the perturbation steps and when the prediction flipped.
    """
    flip_steps = perturbation_results["flip_steps"]
    original_prediction = perturbation_results['original_prediction']

    # The perturbed prediction is the opposite of the original prediction
    perturbed_prediction = 1 - original_prediction

    # Plotting the flip steps as individual points where the prediction flipped
    # Flip steps will be at the opposite value of the original prediction
    plt.plot(flip_steps, [perturbed_prediction] * len(flip_steps), 'ro', label="Prediction Flip")

    # Plotting the original prediction as a constant line at 0 or 1
    plt.axhline(y=original_prediction, color="g", linestyle="--", label="Original Prediction")

    plt.xlabel("Number of Perturbed Features")
    plt.ylabel("Prediction Status")
    plt.title("Perturbation Steps and Prediction Flips")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_perturbation_curve(accuracy_results: Dict[str, list]) -> None:
    """
    Plots the perturbation curve, showing the accuracy at each perturbation step.

    Args:
        accuracy_results (dict): Dictionary containing 'perturbation_steps' and 'accuracies'.
            - 'perturbation_steps' (list): List of perturbation steps (number of features perturbed).
            - 'accuracies' (list): List of accuracy values corresponding to each perturbation step.

    Returns:
        None: Displays a plot showing the accuracy as features are perturbed.
    """
    plt.plot(
        accuracy_results["perturbation_steps"],
        accuracy_results["accuracies"],
        marker="o",
    )
    plt.xlabel("Number of Perturbed Features")
    plt.ylabel("Accuracy")
    plt.title("Perturbation Curve")
    plt.grid(True)
    plt.show()


def plot_feature_importance(
    attributions: np.ndarray, feature_names: List[str], label: str
) -> None:
    """
    Plots the feature importance as a bar chart for the given attributions.

    Args:
        attributions (np.ndarray): Numpy array of attribution values indicating feature importance.
        feature_names (List[str]): List of feature names corresponding to the dataset.
        label (str): The label or category for which the explanation is generated.

    Returns:
        None: Displays a bar plot showing the importance of features for the given label.
    """
    # Sort the features based on the attribution values
    feature_importance = sorted(zip(attributions, feature_names), reverse=True)
    attributions_sorted, feature_names_sorted = zip(*feature_importance)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(attributions_sorted), y=list(feature_names_sorted))
    plt.title(f"LIME Feature Importance for {label}")
    plt.xlabel("Attribution Value")
    plt.ylabel("Feature")
    plt.show()


def visualize_aupc(aupc_dict: dict[str, float], normalize: bool = False) -> None:
    """
    Visualizes the Area Under the Perturbation Curve (AUPC) for various categories using a bar plot.

    Parameters:
    ----------
    aupc_dict : dict[str, float]
        A dictionary where keys represent categories (e.g., TP, TN, FP, FN) and values are the corresponding AUPC values.
    normalize : bool, optional
        If True, scales the AUPC values to the range [0, 1] for visualization, by dividing by the maximum value in the dict.
        Defaults to False.

    Returns:
    -------
    None
        Displays the bar plot for AUPC values.
    """
    categories = list(aupc_dict.keys())
    values = np.array(list(aupc_dict.values()))

    # Normalize if specified
    if normalize:
        max_value = values.max()
        if max_value > 0:  # Avoid division by zero
            values = values / max_value

    # Plotting
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, values, color='skyblue', alpha=0.8, edgecolor='black')

    # Adding value annotations on top of bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.01 if normalize else 0.5),  # Adjust position based on normalization
            f"{value:.2f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    plt.title("AUPC Values Across Categories", fontsize=14, fontweight='bold')
    plt.xlabel("Categories", fontsize=12)
    plt.ylabel("AUPC Values" + (" (Normalized)" if normalize else ""), fontsize=12)
    plt.ylim(0, 1 if normalize else values.max() + 1)  # Adjust y-limit for normalization
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
