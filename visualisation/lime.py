import matplotlib.pyplot as plt
import seaborn as sns

def plot_perturbation_steps(perturbation_results):
    """
    Plots the perturbation steps and shows when the prediction flips.

    Args:
        perturbation_results (dict): Dictionary containing 'flip_steps', 'original_prediction', and 'perturbed_prediction'.
    """
    flip_steps = perturbation_results['flip_steps']

    # Plotting flip steps
    plt.plot(flip_steps, [1] * len(flip_steps), 'ro', label='Prediction Flip')
    plt.axhline(y=1, color='g', linestyle='--', label="Original Prediction")
    plt.xlabel("Number of Perturbed Features")
    plt.ylabel("Prediction Status")
    plt.title("Perturbation Steps and Prediction Flips")
    plt.legend()
    plt.grid(True)
    plt.show()





def plot_perturbation_curve(accuracy_results):
    """
    Plots the perturbation curve.

    Args:
        accuracy_results (dict): Results from perturbing the dataset.
    """
    plt.plot(accuracy_results['perturbation_steps'], accuracy_results['accuracies'], marker='o')
    plt.xlabel("Number of Perturbed Features")
    plt.ylabel("Accuracy")
    plt.title("Perturbation Curve")
    plt.grid(True)
    plt.show()



def plot_feature_importance(attributions, feature_names, label):
    """
    Plot feature importance as a bar chart for the given attributions.

    Args:
        attributions: Numpy array of attribution values.
        feature_names: List of feature names corresponding to the dataset.
        label: The label or category for which the explanation is generated.
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
