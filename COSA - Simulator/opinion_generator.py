import numpy as np

from ipywidgets import (VBox, HBox, Layout, Dropdown, IntSlider, FloatSlider,
                        Checkbox, IntText, Textarea, Output, Text, FloatText,
                        Button, Label) # Added Label

# generate opinion in double precision for all functions
np.float64 = np.float_



# Decorator definition (needed if used)
def specify_widget_params(**param_specs):
    def decorator(func):
        func._widget_specs = param_specs
        return func
    return decorator

@specify_widget_params(
    center = {'widget': FloatSlider, 'min': -1, 'max': 1, 'default': 0.5, 'description': 'Center'},
    one_side = {'widget': FloatSlider, 'min': 0, 'max': 1, 'default': 0.5, 'description': 'One Side'}
)
def symetric_opinion(num_nodes, center=0.0, one_side=0.5):
    if num_nodes % 2 == 0:
        opinion = np.empty(num_nodes)
        opinion[:num_nodes//2] = np.ones(num_nodes//2) * one_side
        opinion[num_nodes // 2:] = -opinion[:num_nodes//2]
        opinion = opinion + center
    else:
        opinion = np.empty(num_nodes)
        opinion[:num_nodes//2] = np.ones(num_nodes//2) * one_side
        opinion[num_nodes // 2] = center
        opinion[num_nodes // 2 + 1:] = -opinion[:num_nodes//2]
        opinion = opinion + center
    return opinion

@specify_widget_params(
    seed = {'widget': IntText, 'default': 0, 'description': 'Seed'}
)
def symetric_random(num_nodes, seed=0):
    np.random.seed(seed)
    if num_nodes % 2 == 0:
        opinion = np.random.rand(num_nodes)
        opinion[num_nodes // 2:] = opinion[num_nodes // 2:] * (-1)
    else:
        opinion = np.random.rand(num_nodes)
        opinion[num_nodes // 2] = 0
        opinion[num_nodes // 2 + 1:] = -opinion[num_nodes // 2 + 1:]
    return opinion

@specify_widget_params(
    opinion = {'widget': Textarea, 'default': '', 'description': 'x1,x2,...'}
)
def manual_opinion(num_nodes, opinion):
    opinion = np.array(opinion.split(','), dtype=float)
    if len(opinion) != num_nodes:
        raise ValueError(f"Length of opinion array ({len(opinion)}) must match num_nodes ({num_nodes})")
    return opinion


# zero opinion
def zero_opinion(num_nodes):
    return np.zeros(num_nodes)

# one opinion
@specify_widget_params(
    value = {'widget': FloatSlider, 'min': -1, 'max': 1, 'default': 0.5, 'description': 'Value'}
)
def identic_opinion(num_nodes, value=0.5):
    return np.ones(num_nodes) * value


# random opinion
@specify_widget_params(
    seed = {'widget': IntText, 'default': 0, 'description': 'Seed'}
)
def random_opinion(num_nodes, seed=0):
    np.random.seed(seed)
    return np.random.rand(num_nodes) * 2 - 1

# eigendecomposition
@specify_widget_params(
    scalar_opinion = {'widget': FloatSlider, 'min': -1, 'max': 1, 'default': 0.5, 'description': 'Scaling'},
    eigenmode = {'widget': IntText, 'default': 0, 'description': 'Eigenmode'},
)
def eigendecomposition_opinion(num_nodes, scalar_opinion=0.5, eigenmode=0, adj_matrix=None):
    # normalize the adj_matrix
    laplacian_matrix = np.diag(np.sum(adj_matrix, axis=1)) - adj_matrix
    normalized_laplacian_matrix = np.diag(1 / np.diag(laplacian_matrix)) @ laplacian_matrix
    # get the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(normalized_laplacian_matrix)
    # sort the eigenvalues and eigenvectors
    eigenvalues_sorted = np.sort(eigenvalues)
    eigenvectors_sorted = eigenvectors[:, np.argsort(eigenvalues)]


    # get the opinion
    opinion = eigenvectors_sorted[:, eigenmode] * scalar_opinion
    opinion = np.real(opinion)
    return opinion

# eigendecomposition
@specify_widget_params(
    scalar_opinion={'widget': FloatSlider, 'min': -1, 'max': 1, 'default': 0.5, 'description': 'Scaling'},
    eigenmode={'widget': IntText, 'default': 0, 'description': 'Eigenmode'},
    seed={'widget': IntText, 'default': 0, 'description': 'Seed'},
    SNR={'widget': FloatSlider, 'min': 0, 'max': 100, 'default': 10, 'description': 'SNR (std ratio)'},
    x_0={'widget': FloatSlider, 'min': -1, 'max': 1, 'default': 0.0, 'description': 'Center'}
)
def eigendecomposition_opinion_with_noise(num_nodes, scalar_opinion=0.5, eigenmode=0, adj_matrix=None, seed=0, SNR=10, x_0=0.0):
    """
    Generates an opinion vector based on an eigenvector of the graph Laplacian,
    with added Gaussian noise controlled by a Signal-to-Noise Ratio (SNR).

    The SNR is defined as the ratio of the standard deviation of the signal (the
    eigenvector component) to the standard deviation of the generated noise.
    """
    if adj_matrix is None:
        # Return a default if no adjacency matrix is provided.
        return np.zeros(num_nodes)

    np.random.seed(seed)
    
    # --- Calculate Graph Laplacian and Eigenvectors ---
    degree = np.sum(adj_matrix, axis=1)
    # Avoid division by zero for isolated nodes.
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_degree = 1.0 / degree
    inv_degree[degree == 0] = 0
    inv_degree_matrix = np.diag(inv_degree)
    
    laplacian_matrix = np.diag(degree) - adj_matrix
    # Using the random-walk normalized Laplacian
    normalized_laplacian_matrix = inv_degree_matrix @ laplacian_matrix
    
    eigenvalues, eigenvectors = np.linalg.eig(normalized_laplacian_matrix)
    
    # Sort eigenvectors based on their eigenvalues
    sorted_indices = np.argsort(eigenvalues)
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    # --- Generate Signal and Noise ---
    
    # 1. The "signal" is the selected eigenvector scaled by the scalar_opinion.
    signal = np.real(eigenvectors_sorted[:, eigenmode]) * scalar_opinion
    
    # 2. Calculate the standard deviation of the signal.
    signal_std = np.std(signal)

    # 3. Calculate the desired standard deviation of the noise.
    # From SNR = signal_std / noise_std  =>  noise_std = signal_std / SNR
    if signal_std < 1e-9:
        # If signal is constant (std is zero), noise cannot be scaled relative to it.
        noise = np.zeros(num_nodes)
    else:
        # Add a small epsilon to avoid division by zero when SNR is 0.
        noise_std = signal_std / (SNR + 1e-9)
        
        # 4. Generate noise with the correct standard deviation.
        noise = (np.random.randn(num_nodes)*2 -1 )* noise_std
    
    # 5. Combine signal, noise, and the center offset.
    opinion = signal + noise + x_0
    
    return np.real(opinion)