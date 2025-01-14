from typing import Tuple, Dict

import torch
from sklearn.metrics import f1_score


def train_model(
    model: torch.nn.Module,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor],
    optimizer_: str,
    criterion: str,
    device: torch.device,
    batch_size: int = 32,
    num_epochs: int = 10,
) -> Dict[str, list]:
    """
    Trains a PyTorch model using manual batching for training and validation datasets.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_data (tuple): A tuple of (inputs, labels) for the training set.
        val_data (tuple): A tuple of (inputs, labels) for the validation set.
        optimizer_ (str): The name of the optimizer class (e.g., 'SGD', 'Adam').
        criterion (str): The loss function name (e.g., 'CrossEntropyLoss').
        device (torch.device): The device (CPU or GPU) to train the model on.
        batch_size (int): Batch size for training. Default is 32.
        num_epochs (int): Number of epochs for training. Default is 10.

    Returns:
        dict: A dictionary containing lists for training and validation losses, accuracies, and F1 scores.
            Keys include 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_f1', 'val_f1'.
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
    }

    model.to(device)
    optimizer = getattr(torch.optim, optimizer_)(model.parameters())  # Get optimizer
    criterion = getattr(torch.nn.functional, criterion)  # Get loss function

    # Training phase
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # Set the model to training mode
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        # For F1 score calculation
        all_preds_train, all_labels_train = [], []

        # Manually batch the training data
        for i in range(0, len(train_data[0]), batch_size):
            inputs = train_data[0][i : i + batch_size]
            labels = train_data[1][i : i + batch_size]

            inputs, labels = torch.tensor(inputs).to(device), torch.tensor(labels).to(
                device
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            all_preds_train.extend(preds.cpu().numpy())
            all_labels_train.extend(labels.cpu().numpy())

        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        # Calculate F1 score for training
        epoch_train_f1 = f1_score(all_labels_train, all_preds_train, average="weighted")

        print(
            f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.4f}"
        )
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["train_f1"].append(epoch_train_f1)

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        all_preds_val, all_labels_val = [], []

        # Manually batch the validation data
        with torch.no_grad():
            for i in range(0, len(val_data[0]), batch_size):
                inputs = val_data[0][i : i + batch_size]
                labels = val_data[1][i : i + batch_size]

                inputs, labels = torch.tensor(inputs).to(device), torch.tensor(
                    labels
                ).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)

                all_preds_val.extend(preds.cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / total
        epoch_val_acc = correct / total

        # Calculate F1 score for validation
        epoch_val_f1 = f1_score(all_labels_val, all_preds_val, average="weighted")

        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_f1"].append(epoch_val_f1)

        print(f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_acc:.4f}")

    return history
