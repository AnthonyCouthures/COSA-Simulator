from widget_creator import get_dynamic_params_from_pool, create_all_widgets
import networkx as nx
import numpy as np
import inspect
from Dynamics import DynamicalSystem
from signal_function import make_signal_jit, make_signal_vector
import ipywidgets as widgets
from ipywidgets import Output, Box, Layout
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from functools import partial

style = """
    <style>
       .jupyter-widgets-output-area .output_scroll {
            height: unset !important;
            border-radius: unset !important;
            -webkit-box-shadow: unset !important;
            box-shadow: unset !important;
        }
        .jupyter-widgets-output-area  {
            height: auto !important;
        }
    </style>
    """

widgets.HTML(style)



def get_graph_from_widgets(full_widgets):
    # print("full_widgets", full_widgets['dynamic_widgets_pool'])
    num_nodes = full_widgets['num_nodes'].value
    graph_generator_widget = full_widgets['graph_generator'] # Get the widget
    graph_generator_func = graph_generator_widget.value # Get the function
    # print('ok')
    selected_func_name = list(graph_generator_widget.options)[list(graph_generator_widget.options.values()).index(graph_generator_func)]
    # print("full_widgets['dynamic_widgets_pool']['graph'].keys()", full_widgets['dynamic_widgets_pool']['graph'].keys())
    dynamic_graph_args = get_dynamic_params_from_pool(
        full_widgets['dynamic_widgets_pool']['graph'], # Access the graph pool
        graph_generator_func,
        selected_func_name
    )

    return graph_generator_func(num_nodes=num_nodes, **dynamic_graph_args)

def get_graph_pos_from_widgets(full_widgets):
    graph = get_graph_from_widgets(full_widgets)
    G = nx.from_numpy_array(graph)
    seed = full_widgets['seed_display_graph'].value
    # print(seed)
    # print(full_widgets['pos'].value)
    if full_widgets['pos'].value == 'random':
        return None
    elif full_widgets['pos'].value == 'circular':
        return nx.circular_layout(G)
    elif full_widgets['pos'].value == 'spring':
        return nx.spring_layout(G, seed=seed)
    elif full_widgets['pos'].value == 'spectral':
        return nx.spectral_layout(G)
    elif full_widgets['pos'].value == 'shell':
        return nx.shell_layout(G)
    elif full_widgets['pos'].value == 'kamada_kawai':
        return nx.kamada_kawai_layout(G)
    elif full_widgets['pos'].value == 'spiral':
        return nx.spiral_layout(G)
    elif full_widgets['pos'].value == 'multipartite':
        return nx.multipartite_layout(G)
    elif full_widgets['pos'].value == 'planar':
        return nx.planar_layout(G)
    else:
        return None
    

def get_graph_pos_from_widgets(full_widgets):
    graph = get_graph_from_widgets(full_widgets)
    G = nx.from_numpy_array(graph)
    position_generator_widget = full_widgets['position_graph']
    position_generator_func = position_generator_widget.value
    position_generator_name = list(position_generator_widget.options)[list(position_generator_widget.options.values()).index(position_generator_func)]
    dynamic_position_args = get_dynamic_params_from_pool(
        full_widgets['dynamic_widgets_pool']['position_graph'], # Access the position pool
        position_generator_func,
        position_generator_name
    )
    return position_generator_func(G, **dynamic_position_args)

def generate_opinion_from_widgets(full_widgets):
    # generate the opinion
    opinion_generator_widget = full_widgets['opinion_generator']
    opinion_generator_func = opinion_generator_widget.value
    opinion_generator_name = list(opinion_generator_widget.options)[list(opinion_generator_widget.options.values()).index(opinion_generator_func)]
    num_nodes = full_widgets['num_nodes'].value
    dynamic_opinion_args = get_dynamic_params_from_pool(
        full_widgets['dynamic_widgets_pool']['opinion'], # Access the opinion pool
        opinion_generator_func,
        opinion_generator_name
    )
    if opinion_generator_func.__name__ == 'eigendecomposition_opinion' or opinion_generator_func.__name__ == 'eigendecomposition_opinion_with_noise':
        adj_matrix = get_graph_from_widgets(full_widgets)
        dynamic_opinion_args['adj_matrix'] = adj_matrix
        opinion = opinion_generator_func(num_nodes=num_nodes, **dynamic_opinion_args)
    else:
        opinion = opinion_generator_func(num_nodes=num_nodes, **dynamic_opinion_args)
    return opinion

def generate_signal_from_widgets(full_widgets):
    # generate the signal
    # print(full_widgets.keys())
    signal_generator_widget = full_widgets['signal_generator']
    x0_signal = full_widgets['x0'].value
    y0_signal = full_widgets['y0'].value
    scalling_signal = full_widgets['scalling'].value
    signal_generator_func = signal_generator_widget.value
    signal_generator_name = list(signal_generator_widget.options)[list(signal_generator_widget.options.values()).index(signal_generator_func)]
    # print("signal_generator_name", signal_generator_name)
    sig = inspect.signature(signal_generator_func)
    param_names = [p.name for p in sig.parameters.values() if p.name not in ('x',)] 
    dynamic_signal_args = get_dynamic_params_from_pool(
        full_widgets['dynamic_widgets_pool']['signal_opinion'], # Access the signal pool
        signal_generator_func,
        signal_generator_name
    )
    dynamic_tuple = tuple(dynamic_signal_args[name] for name in param_names)
    signal_jit = make_signal_jit(
        signal_generator_func,
        x0 = x0_signal,
        y0 = y0_signal,
        scalling = scalling_signal,
        dynamic_args_tuple=dynamic_tuple
    )

    return signal_jit


def get_transform_graph_from_widgets(full_widget):
    transform_graph_widget = full_widget['transform_graph']
    transform_graph_func = transform_graph_widget.value
    transform_graph_name = list(transform_graph_widget.options)[list(transform_graph_widget.options.values()).index(transform_graph_func)]
    dynamic_transform_args = get_dynamic_params_from_pool(
        full_widget['dynamic_widgets_pool']['transform_graph'], # Access the transform pool
        transform_graph_func,
        transform_graph_name
    )
    return partial(transform_graph_func, **dynamic_transform_args)


def get_graph_with_transform_from_widgets(full_widget):
    adj_matrix = get_graph_from_widgets(full_widget)
    transform_graph_func = get_transform_graph_from_widgets(full_widget)
    adj_matrix = transform_graph_func(adj_matrix)
    return adj_matrix

def get_system_from_widgets(full_widget):
    adj_matrix = get_graph_from_widgets(full_widget)
    transform_graph_func = get_transform_graph_from_widgets(full_widget)
    adj_matrix = transform_graph_func(adj_matrix)
    signal = generate_signal_from_widgets(full_widget)
    x0 = generate_opinion_from_widgets(full_widget)
    dt = full_widget['time_step'].value
    method = full_widget['discretization_method'].value
    sys = DynamicalSystem(x0, adj_matrix, dt, signal_jit=signal, method=method)
    return sys


def simulator():

    

    
    output_graph = Output()

    @output_graph.capture(clear_output=True, wait=True)
    def plot_main_simulation_callback():
        # plot the graph
        try:
            adj = get_graph_from_widgets(all_widgets_dict)
            degree = np.sum(adj, axis=1)
            degree_matrix = np.diag(degree)
            laplacian = degree_matrix - adj
            inv_degree_matrix = np.linalg.inv(degree_matrix)
            normalized_laplacian = inv_degree_matrix @ laplacian
            lambda_2 = np.sort(np.linalg.eigvals(normalized_laplacian))[1]
            # print(np.sort(np.linalg.eigvals(normalized_laplacian)))

            
            transform_graph_func = get_transform_graph_from_widgets(all_widgets_dict)
            adj = transform_graph_func(adj)
            
            show_edge = all_widgets_dict['show_edge'].value

            opinion = generate_opinion_from_widgets(all_widgets_dict)
            colors = opinion
            signal = generate_signal_from_widgets(all_widgets_dict)
            signal_vector = make_signal_vector(signal)
            colors_signal = signal_vector(opinion)
            # print(colors_signal)
            # cmap = plt.cm.coolwarm
            cmap = plt.cm.coolwarm
            norm = plt.Normalize(vmin=-1, vmax=1)
            pos = get_graph_pos_from_widgets(all_widgets_dict)
            fig = plt.figure(figsize=(12,6))
            gs = GridSpec(2, 2, width_ratios=[1, 1])
            ax_graph = fig.add_subplot(gs[:, 0])
            #remove the ticks
            ax_graph.set_xticks([])
            ax_graph.set_yticks([])
            ax_signal = fig.add_subplot(gs[1, 1])
            ax_signal_opinion = fig.add_subplot(gs[0, 1])
            ax_signal_opinion.set_ylabel('State')
            ax_signal_opinion.grid(True)
            ax_signal.set_ylabel('Signal')
            ax_signal.grid(True)
            ax_signal.set_xlabel('t')


            sys = get_system_from_widgets(all_widgets_dict)

            num_steps = int(all_widgets_dict['T'].value/all_widgets_dict['time_step'].value)
            time = np.linspace(0, all_widgets_dict['T'].value, num_steps+1)
            trajectory = sys.run_and_return_all(num_steps)
            ax_signal_opinion.plot(time, trajectory)
            signal_trajectory = np.empty_like(trajectory)
            for i in range(all_widgets_dict['num_nodes'].value):
                signal_trajectory[:, i] = signal_vector(trajectory[:, i])
            ax_signal.plot(time, signal_trajectory)

            colors = trajectory[-1, :]
            colors_signal = cmap(norm(colors))

            with output_graph:
                G = nx.from_numpy_array(adj)
                if show_edge:
                    edge_list = list(G.edges())
                    edge_color = []
                    for i,j in edge_list:
                        if adj[i,j] == -1:
                            edge_color.append('red')
                        else:
                            edge_color.append('black')
                    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=0.5, ax=ax_graph)
                
                nx.draw_networkx_nodes(G, node_color=colors_signal, pos=pos, ax=ax_graph, node_size=100)
                
                ax_graph.legend([f'$\\lambda_2 = {lambda_2:.3f}$'], loc='upper right')
                
                plt.show()

            # print(get_graph_with_transform_from_widgets(all_widgets_dict))

        except Exception as e:
            print(f"Error plotting graph: {e}")
            return
        

    all_widgets_dict, all_ui_layout = create_all_widgets(plot_main_simulation_callback)

    # initial the output_graph

    plot_main_simulation_callback()

    all_ui_layout.layout = Layout(
        border='1px solid red',
        width='100%',
        flex_direction='row',
        # align_items='center',
        flex_wrap='wrap'
        )


    all_ui_layout.layout = Layout(
        display='flex',
        flex_flow='row wrap',
        height='40%',
        # width='40%', 
        # align_items='stretch'
        )

    # display(all_ui_layout)
    output_graph.layout = Layout(flex='auto', width='100%', overflow='auto')  
    UI = Box([output_graph, all_ui_layout])

    # make the UI flex
    UI.layout = Layout(flex='1 1 auto', width='100%', flex_flow='flex wrap', height ='auto')


    return (UI, all_widgets_dict)


def extract_selected_parameters(full_widgets):
    """
    Extracts the currently selected parameters from the widgets.

    Args:
        full_widgets (dict): A dictionary containing all the widgets.

    Returns:
        dict: A dictionary of the selected parameters.
    """
    params = {}

    # Extract general parameters
    for key in ['num_nodes', 'seed_display_graph', 'time_step', 'T', 'discretization_method', 'x0', 'y0', 'scalling', 'show_edge']:
        if key in full_widgets:
            params[key] = full_widgets[key].value

    # Graph parameters
    graph_generator_widget = full_widgets['graph_generator']
    graph_generator_func = graph_generator_widget.value
    selected_func_name = list(graph_generator_widget.options.keys())[list(graph_generator_widget.options.values()).index(graph_generator_func)]
    params['graph_generator'] = selected_func_name
    params['graph_params'] = get_dynamic_params_from_pool(
        full_widgets['dynamic_widgets_pool']['graph'],
        graph_generator_func,
        selected_func_name
    )

    # Position parameters
    position_generator_widget = full_widgets['position_graph']
    position_generator_func = position_generator_widget.value
    position_generator_name = list(position_generator_widget.options.keys())[list(position_generator_widget.options.values()).index(position_generator_func)]
    params['position_generator'] = position_generator_name
    params['position_params'] = get_dynamic_params_from_pool(
        full_widgets['dynamic_widgets_pool']['position_graph'],
        position_generator_func,
        position_generator_name
    )

    # Opinion parameters
    opinion_generator_widget = full_widgets['opinion_generator']
    opinion_generator_func = opinion_generator_widget.value
    opinion_generator_name = list(opinion_generator_widget.options.keys())[list(opinion_generator_widget.options.values()).index(opinion_generator_func)]
    params['opinion_generator'] = opinion_generator_name
    params['opinion_params'] = get_dynamic_params_from_pool(
        full_widgets['dynamic_widgets_pool']['opinion'],
        opinion_generator_func,
        opinion_generator_name
    )

    # Signal parameters
    signal_generator_widget = full_widgets['signal_generator']
    signal_generator_func = signal_generator_widget.value
    signal_generator_name = list(signal_generator_widget.options.keys())[list(signal_generator_widget.options.values()).index(signal_generator_func)]
    params['signal_generator'] = signal_generator_name
    params['signal_params'] = get_dynamic_params_from_pool(
        full_widgets['dynamic_widgets_pool']['signal_opinion'],
        signal_generator_func,
        signal_generator_name
    )

    # Transform graph parameters
    transform_graph_widget = full_widgets['transform_graph']
    transform_graph_func = transform_graph_widget.value
    transform_graph_name = list(transform_graph_widget.options.keys())[list(transform_graph_widget.options.values()).index(transform_graph_func)]
    params['transform_graph'] = transform_graph_name
    params['transform_graph_params'] = get_dynamic_params_from_pool(
        full_widgets['dynamic_widgets_pool']['transform_graph'],
        transform_graph_func,
        transform_graph_name
    )

    return params