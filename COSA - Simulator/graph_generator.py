import networkx as nx
import numpy as np

from ipywidgets import (Dropdown, IntSlider, FloatSlider,
                        Checkbox, IntText, Textarea, Output, Text, FloatText,
                        Button, Label) # Added Label

GRAPH_STANDARD_ARGS = {'num_nodes', 'seed'}

# Decorator definition (needed if used)
def specify_widget_params(**param_specs):
    def decorator(func):
        func._widget_specs = param_specs
        return func
    return decorator


@specify_widget_params(
    k={'widget': IntSlider, 'min': 1, 'max': 50, 'default': 4, 'description': 'K'}, 
    seed_generate_k_regular={'widget': IntSlider, 'min': 0, 'max': 1000, 'default': 0, 'description': 'Seed'}
    )
def k_regular_graph(num_nodes: int, k: int = 4, seed_generate_k_regular: int | None = None, **kwargs):
    """Generates a k-regular graph (dummy)."""
    # Dummy implementation for testing widget creation
    # print(f"Generating k-regular with k={k}, seed={seed}")
    if k >= num_nodes:
         # print(f"Warning: k ({k}) >= num_nodes ({num_nodes}). Cannot generate k-regular graph.")
         return np.zeros((num_nodes, num_nodes))
    try:
        G = nx.random_regular_graph(k, num_nodes, seed=seed_generate_k_regular)
        return nx.to_numpy_array(G)
    except nx.NetworkXError:
        print(f"Could not generate {k}-regular graph with {num_nodes} nodes. Check parameters.")
        return np.zeros((num_nodes, num_nodes)) # Return empty on failure
    

    # generate erdos renyi graph
@specify_widget_params(
        prob_connection_erdos_renyi={'widget': FloatSlider, 'min': 0, 'max': 5, 'default': 0.5, 'description': 'P connection'},
        seed_generate_erdos_renyi={'widget': IntSlider, 'min': 0, 'max': 1000, 'default': 0, 'description': 'Seed'},
    )
def generate_erdos_renyi_adj_matrix(num_nodes, prob_connection_erdos_renyi, seed_generate_erdos_renyi, **kwargs):
    if prob_connection_erdos_renyi > 1:
        prob_connection_erdos_renyi = prob_connection_erdos_renyi / num_nodes
    adj_matrix = nx.to_numpy_array(nx.erdos_renyi_graph(num_nodes, prob_connection_erdos_renyi, seed=seed_generate_erdos_renyi))
    return adj_matrix

# generate watts strogatz graph connected
@specify_widget_params(
    k={'widget': IntSlider, 'min': 1, 'max': 50, 'default': 4, 'description': 'K'},
    prob_connection_watts_strogatz={'widget': FloatSlider, 'min': 0, 'max': 1, 'default': 0.5, 'step': 0.01, 'description': 'P connection'},
    seed_generate_watts_strogatz={'widget': IntSlider, 'min': 0, 'max': 1000, 'default': 0, 'description': 'Seed'},
)
def generate_watts_strogatz_adj_matrix(num_nodes, k, prob_connection_watts_strogatz, seed_generate_watts_strogatz, **kwargs):
    adj_matrix = nx.to_numpy_array(nx.watts_strogatz_graph(num_nodes, k, prob_connection_watts_strogatz, seed=seed_generate_watts_strogatz))
    return adj_matrix


def ring_graph(num_nodes: int, **kwargs):
     """Generates a ring graph (dummy)."""
     # print(f"Generating ring graph with seed={seed}")
     G = nx.cycle_graph(num_nodes)
     return nx.to_numpy_array(G)

def star_graph(num_nodes: int, **kwargs):
    """Generates a star graph (dummy)."""
    # print(f"Generating star graph with seed={seed}")
    G = nx.star_graph(num_nodes-1)
    return nx.to_numpy_array(G)

def path_graph(num_nodes: int, **kwargs):
    """Generates a path graph (dummy)."""
    # print(f"Generating path graph with seed={seed}")
    G = nx.path_graph(num_nodes)
    return nx.to_numpy_array(G)

def grid_2d_graph(num_nodes: int, **kwargs):
    """Generates a 2D grid graph (dummy)."""
    # print(f"Generating 2D grid graph with seed={seed}")
    try:
        G = nx.grid_2d_graph(int(np.sqrt(num_nodes)), int(np.sqrt(num_nodes)))
        return nx.to_numpy_array(G)
    except nx.NetworkXError:
        print(f"Could not generate 2D grid graph with {num_nodes} nodes. Check parameters.")
        return np.zeros((num_nodes, num_nodes))

def fan_graph(num_nodes: int, **kwargs):
    """Generates a fan graph (dummy)."""
    # print(f"Generating fan graph with seed={seed}")
    # A simple fan: node 0 connected to all others
    adj = np.zeros((num_nodes, num_nodes))
    if num_nodes > 1:
        adj[0, 1:] = 1
        adj[1:, 0] = 1
    return adj

def complete_graph(num_nodes: int, **kwargs):
    """Generates a complete graph (dummy)."""
    # print(f"Generating complete graph with seed={seed}")
    G = nx.complete_graph(num_nodes)
    return nx.to_numpy_array(G)

def equitable_10_graph(num_nodes: int, seed: int | None = None, **kwargs):
    """Generates a specific 10-node graph (dummy)."""
    # print(f"Generating equitable 10 graph with seed={seed}")
    adj = np.array([
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0, 1, 1, 0]
    ], dtype=float)
    if num_nodes != 10:
        # print(f"Warning: equitable_10_graph requires 10 nodes, got {num_nodes}. Returning 10x10 matrix.")
        # Maybe return a resized version or zeros if size mismatch is an error?
        # For this demo, let's return zeros if size != 10
        # return adj # Original behavior
        print(f"Warning: equitable_10_graph requires exactly 10 nodes. Cannot generate for {num_nodes} nodes.")
        return np.zeros((num_nodes, num_nodes))
    return adj
    

    # --- New: Powerlaw Cluster Graph generator with fixed n nodes
@specify_widget_params(
    m={ 'widget': IntSlider, 'min': 1, 'max': 50, 'step': 1, 'default': 3, 'description': 'M' },
    p_tri={ 'widget': FloatSlider, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.1, 'description': 'p triangle' },
    seed_plg={ 'widget': IntSlider, 'min': 0, 'max': 1000, 'step': 1, 'default': 42, 'description': 'Seed' },
)
def generate_powerlaw_cluster_adj_matrix(
    num_nodes, m, p_tri, seed_plg, **kwargs
):
    """
    Generates an adjacency matrix using the Holme–Kim powerlaw cluster model.

    - n: total nodes
    - m: edges to attach from new node to existing
    - p_tri: probability of adding triangle after random edge
    - seed_plg: RNG seed
    - create_using: if True, use DiGraph, else Graph
    """
    # choose graph type
    G = nx.powerlaw_cluster_graph(
        n=num_nodes,
        m=m,
        p=p_tri,
        seed=seed_plg,
    )
    return nx.to_numpy_array(G)

@specify_widget_params(
    sizes_str={ 'widget': Text, 'default': '50,50', 'description': '(N1,N2,...)' },
    p_within={ 'widget': FloatSlider, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.8, 'description': 'p inner' },
    p_between={ 'widget': FloatSlider, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.2, 'description': 'p outer' },
    seed_sbm={ 'widget': IntSlider, 'min': 0, 'max': 1000, 'step': 1, 'default': 42, 'description': 'Seed' },
)
def generate_sbm_adj_matrix(
    num_nodes, sizes_str,
    p_within, p_between,
    seed_sbm,
    **kwargs
):
    """
    Generates a stochastic block model adjacency matrix with a fixed total number of nodes.

    - `sizes_str` is a comma-separated list of positive numbers (block weights or sizes).
    - These are normalized to sum to `num_nodes` (and then rounded to integers).
    """
    # parse CSV of block weights
    try:
        raw = [float(s) for s in sizes_str.split(',') if s.strip()]
        if len(raw) < 1:
            raise ValueError("At least one block size required.")
    except Exception as e:
        raise ValueError(f"Could not parse sizes_str: {e}")

    # normalize into integer block sizes summing to num_nodes
    proportions = np.array(raw) / np.sum(raw)
    # initial integer sizes via floor
    sizes = np.floor(proportions * num_nodes).astype(int)
    # distribute remainder one by one to largest proportions
    remainder = num_nodes - np.sum(sizes)
    if remainder > 0:
        # get indices sorted by descending fractional part
        fracs = (proportions * num_nodes) - np.floor(proportions * num_nodes)
        for idx in np.argsort(fracs)[::-1][:remainder]:
            sizes[idx] += 1

    # build probability matrix
    num_blocks = len(sizes)
    p_matrix = [ [p_within if i==j else p_between for j in range(num_blocks)]
                 for i in range(num_blocks) ]

    # generate the graph
    G = nx.stochastic_block_model(
        list(sizes), p_matrix,
        seed=seed_sbm,
    )

    # return adjacency matrix
    return nx.to_numpy_array(G)


@specify_widget_params(
    sizes_str={ 'widget': Text, 'default': '50,50', 'description': '(N1,N2,...)' },
    k_str={ 'widget': Text, 'default': '4,4', 'description': '(K1,K2,...)' },
    num_edges_str={ 'widget': Text, 'default': '10,10', 'description': '(E1,E2,...)' },
    seed_glue={ 'widget': IntSlider, 'min': 0, 'max': 1000, 'step': 1, 'default': 42, 'description': 'Seed' },
)
def glue_k_regular_graphs(num_nodes, sizes_str, k_str, num_edges_str, seed_glue, **kwargs):

    # parse CSV of block weights
    try:
        proportions = [float(s) for s in sizes_str.split(',') if s.strip()]
        if len(proportions) < 1:
            raise ValueError("At least one block size required.")
    except Exception as e:
        raise ValueError(f"Could not parse sizes_str: {e}")
    try:
        num_edges = [int(s) for s in num_edges_str.split(',') if s.strip()]
        if len(num_edges) < 1:
            raise ValueError("At least one number of edges required.")
    except Exception as e:
        raise ValueError(f"Could not parse num_edges_str: {e}")
    
    k_str = [int(s) for s in k_str.split(',') if s.strip()]

    # normalize into integer block sizes summing to num_nodes
    proportions = np.array(proportions) / np.sum(proportions)
    # initial integer sizes via floor
    sizes = np.floor(proportions * num_nodes).astype(int)
    # make sure sizes are at least num_nodes
    if np.sum(sizes) < num_nodes:
        sizes[0] += num_nodes - np.sum(sizes)
    # make sure sizes it at most num_nodes
    if np.sum(sizes) > num_nodes:
        sizes[0] -= np.sum(sizes) - num_nodes

    for i in range(len(sizes)):
        if sizes[i] < k_str[i]:
            sizes[i] = k_str[i]

    list_of_graphs = []
    for i in range(len(sizes)):
        list_of_graphs.append(nx.random_regular_graph(k_str[i], sizes[i], seed=seed_glue))
    G = nx.disjoint_union_all(list_of_graphs)
    # add some random edges between the two graphs
    np.random.seed(seed_glue)
    for i in range(1, len(list_of_graphs)):
        idx1 = np.random.randint(0, sizes[i-1], num_edges[i-1])
        idx2 = np.random.randint(sizes[i-1], sizes[i-1] + sizes[i], num_edges[i-1])
        G.add_edges_from(list(zip(idx1, idx2)))
    return nx.to_numpy_array(G)
    

@specify_widget_params(
    sizes_str={ 'widget': Text, 'default': '50,50', 'description': '(N1,N2,...)' },
    k_str={ 'widget': Text, 'default': '4,4', 'description': '(K1,K2,...)' },
    num_edges_str={ 'widget': Text, 'default': '10,10', 'description': '(E1,E2,...)' },
    seed_glue={ 'widget': IntSlider, 'min': 0, 'max': 1000, 'step': 1, 'default': 42, 'description': 'Seed' },
)
def glue_k_regular_graphs(num_nodes, sizes_str, k_str, num_edges_str, seed_glue, **kwargs):
    # --- parse inputs (unchanged) ---
    proportions = [float(s) for s in sizes_str.split(',') if s.strip()]
    k_list      = [int(s)   for s in k_str.split(',')       if s.strip()]
    num_edges   = [int(s)   for s in num_edges_str.split(',')if s.strip()]
    np.random.seed(seed_glue)

    # normalize sizes to sum to num_nodes
    props = np.array(proportions)
    props = props / props.sum()
    sizes = np.floor(props * num_nodes).astype(int)
    # adjust first block to fix rounding
    diff = num_nodes - sizes.sum()
    sizes[0] += diff

    # ensure each block is at least k_i
    for i, k_i in enumerate(k_list):
        if sizes[i] < k_i:
            sizes[i] = k_i

    # build each k-regular block
    blocks = [
        nx.random_regular_graph(k_list[i], sizes[i], seed=seed_glue)
        for i in range(len(sizes))
    ]
    G = nx.disjoint_union_all(blocks)

    # compute the start‐index of each block in the union
    offsets = np.concatenate(([0], np.cumsum(sizes)))  # len = n_blocks+1

    # now “glue” each block i to a random other block j ≠ i
    n_blocks = len(sizes)
    for i in range(n_blocks):
        # skip block 0 if you don’t want it to initiate edges; otherwise include i=0 too
        if num_edges[i] <= 0:
            continue

        # choose a random other block
        others = list(range(n_blocks))
        others.remove(i)
        j = np.random.choice(others)

        # sample endpoints within block i and block j
        start_i, end_i = offsets[i], offsets[i+1]
        start_j, end_j = offsets[j], offsets[j+1]

        u_nodes = np.random.randint(start_i, end_i, size=num_edges[i])
        v_nodes = np.random.randint(start_j, end_j, size=num_edges[i])
        G.add_edges_from(zip(u_nodes, v_nodes))

    return nx.to_numpy_array(G)


# bipartite graph
@specify_widget_params(
    prop_nodes_bipartite={'widget': FloatSlider, 'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.01, 'description': 'Ratio First'},
    prob_connection_bipartite={'widget': FloatSlider, 'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.01, 'description': 'P connection'},
    seed_bipartite={'widget': IntSlider, 'min': 0, 'max': 1000, 'default': 0, 'description': 'Seed'},
)
def bipartite_graph(num_nodes, prop_nodes_bipartite, prob_connection_bipartite, seed_bipartite, **kwargs):
    num_nodes_bipartite = int(num_nodes * prop_nodes_bipartite)
    num_nodes_bipartite_2 = num_nodes - num_nodes_bipartite
    G = nx.bipartite.random_graph(num_nodes_bipartite, num_nodes_bipartite_2, prob_connection_bipartite, seed=seed_bipartite)
    return nx.to_numpy_array(G)


##### POSITION GENERATORS #####

POSITION_STANDARD_ARGS = {'G'}

@specify_widget_params(
    k_spring={'widget': IntSlider, 'min': 1, 'max': 50, 'default': 1, 'description': 'K'},
    iters={'widget': IntSlider, 'min': 1, 'max': 500, 'default': 50, 'description': 'Iters'},
    thresh={'widget': FloatSlider, 'min': 1, 'max': 100, 'default': 10, 'description': 'Thresh'},
    seed_spring={'widget': IntSlider, 'min': 0, 'max': 1000, 'default': 0, 'description': 'Seed'},
)
def spring_layout(G, k_spring, iters, thresh, seed_spring):
    return nx.spring_layout(G, k=k_spring * 1/ np.sqrt(G.number_of_nodes()), iterations=iters, threshold=thresh * 1e-5, seed=seed_spring)

def circular_layout(G):
    return nx.circular_layout(G)

def spectral_layout(G):
    return nx.spectral_layout(G)

@specify_widget_params(
    max_iter_force_atlas={'widget': IntSlider, 'min': 1, 'max': 50, 'default': 100, 'description': 'Max Iters'},
    jitter_tolerance_force_atlas={'widget': FloatSlider, 'min': 0.0001, 'max': 1, 'default': 1.0, 'step': 0.0001, 'description': 'Jitter Tol'},
    scaling_ratio_force_atlas={'widget': FloatSlider, 'min': 0.0001, 'max': 1, 'default': 2.0, 'step': 0.0001, 'description': 'Scaling R'},
    gravity_force_atlas={'widget': FloatSlider, 'min': 0.0001, 'max': 1, 'default': 1.0, 'step': 0.0001, 'description': 'G'},
    distributed_action_force_atlas={'widget': Checkbox, 'default': False},
    strong_gravity_force_atlas={'widget': Checkbox, 'default': False},
    weight_force_atlas={'widget': FloatSlider, 'min': 0.0001, 'max': 1, 'default': 1.0, 'step': 0.0001, 'description': 'W'},
    seed_force_atlas={'widget': IntSlider, 'min': 0, 'max': 1000, 'default': 0, 'description': 'Seed'},
)
def force_atlas_layout(G, max_iter_force_atlas, jitter_tolerance_force_atlas, scaling_ratio_force_atlas, gravity_force_atlas, distributed_action_force_atlas, strong_gravity_force_atlas, weight_force_atlas, seed_force_atlas):
    return nx.forceatlas2_layout(G, max_iter=max_iter_force_atlas, jitter_tolerance=jitter_tolerance_force_atlas, scaling_ratio=scaling_ratio_force_atlas, gravity=gravity_force_atlas, distributed_action=distributed_action_force_atlas, strong_gravity=strong_gravity_force_atlas, weight=weight_force_atlas, seed=seed_force_atlas)

def kamada_kawai_layout(G):
    return nx.kamada_kawai_layout(G)


def shell_layout(G):
    return nx.shell_layout(G)




# GRAPH TRANSFORMATIONS

def no_transformation(adj_matrix):
    return adj_matrix

@specify_widget_params(
    p_antagonist={'widget': FloatSlider, 'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.01, 'description': 'P antagonist'},
    seed_antagonist={'widget': IntSlider, 'min': 0, 'max': 1000, 'default': 0, 'description': 'Seed'},
)
def random_antagonist_edges(adj_matrix, p_antagonist, seed_antagonist, **kwargs):
    # randomly change the sign of the edges
    np.random.seed(seed_antagonist)
    # must return a symmetric matrix
    for i in range(adj_matrix.shape[0]):
        for j in range(i+1, adj_matrix.shape[1]):
            if np.random.rand() < p_antagonist:
                adj_matrix[i, j] = -adj_matrix[i, j]
                adj_matrix[j, i] = -adj_matrix[j, i]

    
    
    return adj_matrix

@specify_widget_params(
    method_frontier={'widget': Dropdown, 'options': ['Greedy Modularity', 'Louvain'], 'default': 'Greedy Modularity', 'description': 'Method'},
)
def frontier_antagonist_edges(adj_matrix, method_frontier):
    # change the sign of the edges on the frontier between two communities
    # first, find the communities
    if method_frontier == 'Greedy Modularity':
        communities = nx.community.greedy_modularity_communities(G)
    elif method_frontier == 'Louvain':
        communities = nx.community.louvain_communities(G)
    # then, change the sign of the edges on the frontier between the communities
    for i in range(len(communities)):
        for j in range(i+1, len(communities)):
            adj_matrix[communities[i], communities[j]] = -adj_matrix[communities[i], communities[j]]
    return adj_matrix

















