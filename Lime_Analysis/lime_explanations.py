
from captum.attr import Lime


from Lime_Analysis.utils import measure_time, absolute_max_scaling


def initialize_lime(model):
    """
    Initialize the Lime instance for the given model.
    """
    return Lime(model)


@measure_time
def generate_explanations(lime, test_data, true_labels, selected_indices, n_samples=50):
    """
    Generate and visualize explanations for selected indices in tabular data.

    Args:
        lime: Initialized Lime instance.
        test_data: Torch tensor of test data.
        true_labels: Torch tensor of true labels.
        selected_indices: Dictionary with labels (e.g., TP, FP, FN, TN) as keys and list of indices as values.
        n_samples: Number of samples to generate for LIME.

    Returns:
        attribution_dict: Dictionary where keys are labels and values are the corresponding attributions for each selected instance.
    """
    attribution_dict = {}  # Initialize the dictionary to store attributions for each label
    for label, indices in selected_indices.items():
        if len(indices) > 0:  # Ensure at least one instance exists for the category
            idx = indices[0]  # Select the first instance for the given label
            input_tensor = test_data[idx].unsqueeze(0).requires_grad_()  # Add batch dimension
            target = true_labels[idx].item()

            # Generate LIME attributions
            attributions = lime.attribute(input_tensor, target=target, n_samples=n_samples)
            attributions_np = attributions.squeeze().detach().numpy()  # Convert to numpy for easier visualization

            # Store the attributions in the dictionary
            attribution_dict[label] = {
                'indices': indices,
                'attributions': absolute_max_scaling(attributions_np),
            }



    return attribution_dict




