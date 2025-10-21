# __init__.py

# ensure the package root is on sys.path
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# metadata
__version__ = '0.1.0'                   # package version :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
__author__ = 'Anthony Couthures'        # author :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
__description__ = (
    'A simulator for opinion dynamics on networks with environmental feedback'
)                                        # description :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}

# graph generators & layouts :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
from .graph_generator import (
    k_regular_graph,
    generate_erdos_renyi_adj_matrix,
    generate_watts_strogatz_adj_matrix,
    ring_graph,
    fan_graph,
    equitable_10_graph,
    generate_powerlaw_cluster_adj_matrix,
    generate_sbm_adj_matrix,
    glue_k_regular_graphs,
    spring_layout,
    circular_layout,
    spectral_layout,
    force_atlas_layout,
    kamada_kawai_layout,
    shell_layout,
    no_transformation,
    random_antagonist_edges,
    frontier_antagonist_edges,
    star_graph,
    path_graph,
    grid_2d_graph,
    complete_graph,
)

# signal functions & JIT utilities :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
from .signal_function import (
    signal_identity,
    signal_k_linear,
    signal_tanh,
    signal_arctan,
    signal_smooth_step,
    signal_deadzone,
    signal_discontinuity_tanh,
    signal_sign,
    signal_sinus,
    step_func,
    smooth_step_func,
    make_signal_jit,
    make_signal_vector,
)

# opinion generators :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
from .opinion_generator import (
    symetric_opinion,
    symetric_random,
    manual_opinion,
    zero_opinion,
    identic_opinion,
    random_opinion,
)

# widget constructors :contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}:contentReference[oaicite:15]{index=15}
from .widget_creator import (
    create_graph_widgets,
    create_signal_widgets,
    create_opinion_widgets,
    create_simulation_widgets,
    create_all_widgets,
    get_dynamic_params_from_pool,
)

# core dynamics class :contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17}
from .Dynamics import DynamicalSystem

# high‑level simulator entrypoint :contentReference[oaicite:18]{index=18}:contentReference[oaicite:19]{index=19}
from .Simulator import (
    simulator,
    get_graph_from_widgets,
    get_graph_pos_from_widgets,
    generate_opinion_from_widgets,
    generate_signal_from_widgets,
    get_transform_graph_from_widgets,
    get_graph_with_transform_from_widgets,
    get_system_from_widgets,
)

# notebook‑style interactive plot :contentReference[oaicite:20]{index=20}:contentReference[oaicite:21]{index=21}
from .output import (
    graph_widgets_dict,
    graph_ui_layout,
    signal_widgets_dict,
    signal_ui_layout,
    opinion_widgets_dict,
    opinion_ui_layout,
    interactive_plot,
)

__all__ = [
    # graph generation
    'k_regular_graph', 'generate_erdos_renyi_adj_matrix', 'generate_watts_strogatz_adj_matrix',
    'ring_graph', 'fan_graph', 'equitable_10_graph',
    'generate_powerlaw_cluster_adj_matrix', 'generate_sbm_adj_matrix', 'glue_k_regular_graphs',
    'spring_layout', 'circular_layout', 'spectral_layout', 'force_atlas_layout',
    'kamada_kawai_layout', 'shell_layout', 'no_transformation',
    'random_antagonist_edges', 'frontier_antagonist_edges',
    'star_graph', 'path_graph', 'grid_2d_graph', 'complete_graph',

    # signals
    'signal_identity', 'signal_k_linear', 'signal_tanh', 'signal_arctan',
    'signal_smooth_step', 'signal_deadzone', 'signal_discontinuity_tanh',
    'signal_sign', 'signal_sinus', 'step_func', 'smooth_step_func',
    'make_signal_jit', 'make_signal_vector',

    # opinions
    'symetric_opinion', 'symetric_random', 'manual_opinion',
    'zero_opinion', 'identic_opinion', 'random_opinion',

    # widgets
    'create_graph_widgets', 'create_signal_widgets', 'create_opinion_widgets',
    'create_simulation_widgets', 'create_all_widgets', 'get_dynamic_params_from_pool',

    # core dynamics & simulation
    'DynamicalSystem',
    'simulator', 'get_graph_from_widgets', 'get_graph_pos_from_widgets',
    'generate_opinion_from_widgets', 'generate_signal_from_widgets',
    'get_transform_graph_from_widgets', 'get_graph_with_transform_from_widgets',
    'get_system_from_widgets',

    # interactive output
    'graph_widgets_dict', 'graph_ui_layout',
    'signal_widgets_dict', 'signal_ui_layout',
    'opinion_widgets_dict', 'opinion_ui_layout',
    'interactive_plot',
]
