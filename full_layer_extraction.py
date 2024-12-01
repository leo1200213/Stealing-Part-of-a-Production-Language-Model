import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Query the Model with Multiple Inputs to Get Logits

def get_Q(tokenized_prompts, model):
    """
    Query the model with multiple inputs to get logits and construct matrix Q.
    
    Args:
        tokenized_prompts (list): List of tokenized inputs.
        model: The language model.
    
    Returns:
        np.ndarray: Logits matrix Q (l x n).
    """
    logits_list = []
    for tokenized_input in tqdm(tokenized_prompts, desc="Querying model"):
        with torch.no_grad():
            output = model.generate(
                input_ids=tokenized_input,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True
            )
            logits = output.scores[0].squeeze(0).detach().cpu().numpy()
        logits_list.append(logits)
    return np.array(logits_list).T  # Return as matrix Q (l x n)

def dim_extraction(Q):
    """
    Extract the hidden dimensionality of the model using SVD.
    
    Args:
        Q (np.ndarray): Logits matrix Q.
    
    Returns:
        tuple: U matrix, singular values S, and estimated hidden dimension h.
    """
    U, S, _ = torch.linalg.svd(torch.tensor(Q, dtype=torch.float64), full_matrices=False)
    log_singular_values = torch.log(S.abs())
    diffs = log_singular_values[:-1] - log_singular_values[1:]
    filtered_diffs = torch.where(log_singular_values[:-1] > 0, 1, -1) * diffs
    h = torch.argmax(filtered_diffs).item() + 1

    # Plot the singular values
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(S) + 1), S.cpu().numpy(), label='Singular Values')
    plt.ylim(0, 1000)  # Limit the y-axis to values between 0 and 1000
    plt.xlabel('Index of Singular Values')
    plt.ylabel('Singular Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('singular_values_plot.png')
    plt.show()

    return U, S, h

def layer_extraction(U, S, h):
    """
    Extract the final layer weights approximation.
    
    Args:
        U (torch.Tensor): Unitary matrix from SVD.
        S (torch.Tensor): Singular values from SVD.
        h (int): Estimated hidden dimensionality.
    
    Returns:
        torch.Tensor: Reconstructed weight matrix W_tilde.
    """
    U_h = U[:, :h]  # Select the first h columns of U
    S_h = torch.diag(S[:h])  # Construct the diagonal matrix using the first h singular values

    W_tilde = U_h @ S_h
    
    return W_tilde
