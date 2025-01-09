from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

def getBreastData():
    """
    Loads and returns the Breast Cancer Data, properly split into
    Training, Test, and Validation sets, with feature normalization.
    Returns the data in the form of tuples that can be used directly in the training function.
    """
    # Load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Split train+val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit on train data and transform
    X_val = scaler.transform(X_val)         # Transform validation data
    X_test = scaler.transform(X_test)       # Transform test data

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Return as tuples (features, labels)
    train_data = (X_train_tensor, y_train_tensor)
    val_data = (X_val_tensor, y_val_tensor)
    test_data = (X_test_tensor, y_test_tensor)

    return train_data, val_data, test_data, feature_names
