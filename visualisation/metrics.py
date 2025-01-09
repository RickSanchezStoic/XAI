import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy(history):
    # Set the seaborn style for better visuals
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot training accuracy with line style and markers
    plt.plot(history["train_acc"], label='Training Accuracy', color='b', marker='o', linestyle='-', linewidth=2, markersize=6)

    # Plot validation accuracy with line style and markers
    plt.plot(history["val_acc"], label="Validation Accuracy", color='r', marker='s', linestyle='--', linewidth=2, markersize=6)

    # Add title, labels, and grid
    plt.title("Training and Validation Accuracy Over Epochs", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Add a legend to the plot
    plt.legend(loc="lower right", fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()




def plot_f1_scores(history):
    # Set the seaborn style
    sns.set(style="whitegrid")

    # Create the plot for F1 scores
    plt.figure(figsize=(10, 6))

    # Plot training F1 score with line style and markers
    plt.plot(history["train_f1"], label="Training F1 Score", color='b', marker='o', linestyle='-', linewidth=2, markersize=6)

    # Plot validation F1 score with line style and markers
    plt.plot(history["val_f1"], label="Validation F1 Score", color='r', marker='s', linestyle='--', linewidth=2, markersize=6)

    # Add title, labels, and grid
    plt.title("Training and Validation F1 Score Over Epochs", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Add a legend to the plot
    plt.legend(loc="lower right", fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()


import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import seaborn as sns


def plot_roc_auc(model, X_test_tensor, y_test_tensor, device):
    """
    Plots the ROC curve for a given model on test data.

    Args:
        model (torch.nn.Module): The trained model.
        X_test_tensor (torch.Tensor): The input test features.
        y_test_tensor (torch.Tensor): The true test labels.
        device (torch.device): Device on which the model is located (CPU or GPU).
    """
    model.eval()  # Set the model to evaluation mode
    y_true = y_test_tensor.cpu().numpy()  # Convert labels to numpy array for ROC calculation
    y_probs = []  # To store predicted probabilities

    with torch.no_grad():
        # Move the test data to the same device as the model
        X_test_tensor = X_test_tensor.to(device)
        outputs = model(X_test_tensor)

        # For binary classification, use sigmoid to get probabilities for class 1 (positive class)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Assuming binary classification

        # Store the probabilities for ROC calculation
        y_probs = probs.cpu().numpy()

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    # Calculate AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate (Recall)', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.show()

    return roc_auc
