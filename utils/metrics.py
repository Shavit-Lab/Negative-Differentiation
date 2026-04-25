import numpy as np
import torch
from tqdm.contrib.concurrent import process_map
from scipy.stats import wasserstein_distance

def generate_pairs(total_samples, num_pairs):
    assert num_pairs <= 2 * total_samples
    sampled_indices = np.random.choice(total_samples, 2 * num_pairs, replace=False)
    indices1 = sampled_indices[:num_pairs]
    indices2 = sampled_indices[num_pairs:]
    return indices1, indices2

# Compute N_x (maximum L2 norm between input pairs)
def calculate_Nx(inputs, indices1, indices2): # in this case, inputs shape is (131072, 2048)
    # pairwise_distances = pdist(inputs) # this should be used in the case that you want to use all possible input pairs
    pairwise_distances = np.linalg.norm(inputs[indices1] - inputs[indices2], axis=1) # shape is (num_pairs, )
    N_x = np.max(pairwise_distances)  # The maximum L2 norm
    return pairwise_distances, N_x

# Comput N_y (median L2 norm between output pairs)
def calculate_Ny(outputs, indices1, indices2): # in this case, outputs shape is (131072,)
    # pairwise_distances = pdist(outputs)
    pairwise_distances = np.abs(outputs[indices1] - outputs[indices2]) # just calculates the absolute value since outputs are scalar values
    N_y = np.median(pairwise_distances)
    return pairwise_distances, N_y

def calculate_neuron_MD(normalized_input_distances, neuron_outputs, indices1, indices2):
    output_distances, Ny = calculate_Ny(neuron_outputs, indices1, indices2)
    normalized_output_distances = output_distances / Ny
    MD = np.mean(normalized_output_distances / normalized_input_distances)
    return MD

def calculate_MD(inputs, outputs, indices1, indices2):
    input_distances, Nx = calculate_Nx(inputs, indices1, indices2)
    normalized_input_distances = input_distances / Nx
    num_neurons = outputs.shape[1]
    normalized_input_distances_expanded = np.tile(normalized_input_distances, (num_neurons, 1))
    indices1_expanded = np.tile(indices1, (num_neurons, 1))
    indices2_expanded = np.tile(indices2, (num_neurons, 1))
    MDs = process_map(calculate_neuron_MD, normalized_input_distances_expanded, outputs.T, indices1_expanded, indices2_expanded, max_workers=32, chunksize=1)
    return MDs

def normalize_data_columnwise(X):
    if isinstance(X, torch.Tensor):
        X_numpy = X.numpy()  # Convert tensor to numpy array
        return (X_numpy - np.mean(X_numpy, axis=0)) / np.std(X_numpy, axis=0)
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def calculate_neuron_WD(neuron_outputs, standard_normal=None, normalized=False):
    num_samples = neuron_outputs.shape[0]
    if standard_normal is None:
        standard_normal = np.random.normal(0, 1, num_samples) # shape (number_tokens, )
    if normalized:
        return wasserstein_distance(neuron_outputs, standard_normal)
    else:
        normalized_outputs = (neuron_outputs - np.mean(neuron_outputs)) / np.std(neuron_outputs)
        return wasserstein_distance(normalized_outputs, standard_normal)

def calculate_WD(neuron_outputs, standard_normal=None):
    num_samples = neuron_outputs.shape[0]
    num_neurons = neuron_outputs.shape[1]
    normalized_outputs = normalize_data_columnwise(neuron_outputs)
    if standard_normal is None:
        standard_normal = np.random.normal(0, 1, num_samples) # shape (num_samples, )
    standard_normal_expanded = np.tile(standard_normal, (num_neurons, 1))
    normalized=True
    normalized_expanded = np.tile(normalized, (num_neurons, 1))
    WDs = process_map(calculate_neuron_WD, normalized_outputs.T, standard_normal_expanded, normalized_expanded, max_workers=32, chunksize=1)
    return WDs