import inspect
from ipywidgets import (VBox, HBox, Box, Layout, Dropdown, IntSlider, FloatSlider,
                        Checkbox, IntText, Textarea, Output, Text, FloatText,
                        Button, Label, BoundedFloatText, BoundedIntText, Widget) # Added Label

import numpy as np
import matplotlib.pyplot as plt

from signal_function import *
from graph_generator import *
from opinion_generator import *

from IPython.display import display, Math, Latex


card_layout = Layout(
    border='1px solid #ddd', 
    padding='5px', 
    margin='0 0 5px 0', 
    display='flex', 
    flex_flow='column'
)

small_card_layout = Layout(
    border='1px solid #ddd', 
    padding='2px', 
    margin='0 0 2px 0', 
    display='flex', 
    flex_flow='column'
)

def _create_widget_for_param(param, func_specs):
    """Dynamically creates a widget for a given function parameter."""
    name = param.name
    spec = func_specs.get(name, {}) # Get hints from decorator if available

    # Determine Widget Type
    widget_class = spec.get('widget')
    default_val = spec.get('default', param.default)
    value = default_val if default_val is not inspect.Parameter.empty else None

    # Infer widget type if not specified by decorator
    if widget_class is None:
        if param.annotation is int or isinstance(value, int):
            widget_class = IntSlider
        elif param.annotation is float or isinstance(value, float):
            widget_class = FloatSlider
        elif param.annotation is bool or isinstance(value, bool):
            widget_class = Checkbox
        elif param.annotation is list:
            widget_class = Dropdown
        else: # Default to Text for strings or unknown
            widget_class = Text # Or better, FloatText/IntText if type hint exists

    # Create Widget 
    widget_args = {}
    if 'description' in spec: widget_args['description'] = spec['description']
    else: widget_args['description'] = name.replace('_', ' ').title()
    if value is not None:
        widget_args['value'] = value

    # Add min/max/step if specified by decorator
    if 'min' in spec: widget_args['min'] = spec['min']
    if 'max' in spec: widget_args['max'] = spec['max']
    if 'step' in spec: widget_args['step'] = spec['step']
    if 'readout_format' in spec: widget_args['readout_format'] = spec['readout_format']
    if 'options' in spec: widget_args['options'] = spec['options']

    # Sliders typically benefit from continuous_update=False
    if widget_class in (IntSlider, FloatSlider):
        widget_args['continuous_update'] = False

    try:
        # Handle potential edge cases like default=None but type hint is int/float
        if widget_class in (IntSlider, FloatSlider, IntText, FloatText) and value is None:
             # Provide a default value if param is required but default is None
             print(f"Warning: Parameter '{name}' for widget {widget_class.__name__} has default None. Setting value to 0.")
             widget_args['value'] = 0 # Or raise error?


        widget = widget_class(**widget_args)
        
        widget.layout = Layout(width='auto', max_width='300px', min_width='150px')
    except Exception as e:
         print(f"Error creating {widget_class.__name__} for '{name}' with args {widget_args}: {e}")
         widget = Text(description=name, value=str(value) if value is not None else '') # Fallback

    # Initially hide all dynamic widgets
    widget.layout.display = 'none'

    return widget



# --- GRAPH WIDGETS ---

# --- Function Selection & Dynamic Area ---
GRAPH_GENERATORS_UNSORTED = {
    'K-Regular': k_regular_graph,
    'Ring': ring_graph,
    'Fan': fan_graph,
    'Equitable 10': equitable_10_graph,
    'Erdos Renyi': generate_erdos_renyi_adj_matrix,
    'Watts Strogatz': generate_watts_strogatz_adj_matrix,
    'Powerlaw Cluster': generate_powerlaw_cluster_adj_matrix,
    'Stochastic Block Model': generate_sbm_adj_matrix,
    'Glue K-Regular': glue_k_regular_graphs,
    'Star': star_graph,
    'Path': path_graph,
    'Grid 2D': grid_2d_graph,
    'Bipartite': bipartite_graph,
}

GRAPH_GENERATORS = {key: value for key, value in sorted(GRAPH_GENERATORS_UNSORTED.items(),key=lambda x:x[0].lower())}


POSITION_GENERATORS_UNSORTED = {
    'Spring': spring_layout,
    'Circular': circular_layout,
    'Spectral': spectral_layout,
    'Force Atlas': force_atlas_layout,
    'Kamada Kawai': kamada_kawai_layout,
    'Shell': shell_layout,
}

POSITION_GENERATORS = {key: value for key, value in sorted(POSITION_GENERATORS_UNSORTED.items(),key=lambda x:x[0].lower())}
GRAPH_TRANSFORMATIONS = {
    'No Transformation': no_transformation,
    'Random Antagonist Edges': random_antagonist_edges,
    'Frontier Antagonist Edges': frontier_antagonist_edges,
}

GRAPH_STANDARD_ARGS = {'num_nodes', 'seed'}

POSITION_STANDARD_ARGS = {'G'}

TRANSFORMATION_STANDARD_ARGS = {'adj_matrix'}

def generate_dynamic_widgets_pool(
    main_widget : Widget, 
    DICT_OF_FUNCTIONS : dict, DICT_OF_STANDARD_ARGS : dict, 
    main_callback : callable,
    other_callback : callable = None
    ) -> tuple[Widget, VBox, dict]:
    dynamic_widgets_pool = {}
    dynamic_params_container = VBox([])

    widgets_to_add_to_pool = []
    for func_name, func in DICT_OF_FUNCTIONS.items():
        if callable(func):
            sig = inspect.signature(func)
            specs = getattr(func, '_widget_specs', {}) # Get hints

            dynamic_widgets_pool[func_name] = {} # Create entry for this function

            for param_name, param in sig.parameters.items():
                if param_name in DICT_OF_STANDARD_ARGS or \
                   param.kind == param.VAR_POSITIONAL or \
                   param.kind == param.VAR_KEYWORD:
                    continue # Skip standard args and var args

                # Create the widget for this specific parameter
                # _create_widget_for_param already sets layout.display = 'none'
                widget = _create_widget_for_param(param, specs)

            
                # Store the widget in our pool dictionary
                dynamic_widgets_pool[func_name][param_name] = widget

                # Add it to the list to be added to the container
                widgets_to_add_to_pool.append(widget)

    # Add all pre-created widgets to the main dynamic container
    dynamic_params_container.children = tuple(widgets_to_add_to_pool)

    def update_dynamic_widgets_visibility(change):
        selected_func = main_widget.value
        # get the name of the function from the value
        selected_func_name = list(DICT_OF_FUNCTIONS.keys())[list(DICT_OF_FUNCTIONS.values()).index(selected_func)]


        # Iterate through ALL dynamic widgets in the pool
        for func_widgets_map in dynamic_widgets_pool.values():
             for param_widget in func_widgets_map.values():
                 param_widget.layout.display = 'none' # Hide all first

        # Show widgets for the currently selected function
        if selected_func_name in dynamic_widgets_pool.keys():
            for param_widget in dynamic_widgets_pool[selected_func_name].values():
                param_widget.layout.display = 'flex'

        # --- Trigger plot update AFTER function selection changes ---
        if main_callback:
             main_callback()
        if other_callback:
             other_callback()
        # else:
        #     print("Plot callback not set yet during dynamic update.") # Debug

    # Attach observer to the function dropdown to manage visibility
    main_widget.observe(update_dynamic_widgets_visibility, names='value')

    def dynamic_widget_observer(change):
        if main_callback:
            main_callback()
        if other_callback:
            other_callback()


    for func_name, func_widgets_map in dynamic_widgets_pool.items():
         for param_name, widget in func_widgets_map.items():
             # Only observe widgets that actually have a value attribute that changes
             if hasattr(widget, 'value'):
                widget.observe(dynamic_widget_observer, names='value')

    # --- Attach Observers to Static Widgets ---
    static_widgets_to_observe = [
        main_widget,
    ]
    for w in static_widgets_to_observe:
        w.observe(update_dynamic_widgets_visibility, names='value')


    update_dynamic_widgets_visibility({'new': main_widget.value}) # Pass dummy change


    return main_widget, dynamic_params_container, dynamic_widgets_pool




def create_graph_widgets(main_simulation_callback=None):
    """
    Creates graph widgets, including pre-created dynamic parameter widgets
    which are hidden/shown based on graph type selection.
    Uses the Manual Observation pattern for triggering plots.
    """

    Title = Label(value='Graph', style={'font_weight': 'bold'})
    # --- Static Widgets ---
    num_nodes_widget = IntSlider(value=100, min=1, max=500, description='N', continuous_update=False)
    show_edge_widget = Checkbox(value=True, description='Show Edges')

    # Make the graph generator widget
    graph_generator_widget = Dropdown(
        options=GRAPH_GENERATORS,
        value=k_regular_graph,
        description='Generator'
    )

    graph_generator_widget, graph_generator_dynamic_params_container, graph_generator_dynamic_widgets_pool = generate_dynamic_widgets_pool(graph_generator_widget, GRAPH_GENERATORS, GRAPH_STANDARD_ARGS, main_simulation_callback)

    
    # Make the position generator widget
    position_generator_widget = Dropdown(
        options=POSITION_GENERATORS,
        value=spring_layout,
        description='Display'
    )

    position_generator_widget, position_generator_dynamic_params_container, position_generator_dynamic_widgets_pool = generate_dynamic_widgets_pool(position_generator_widget, POSITION_GENERATORS, POSITION_STANDARD_ARGS, main_simulation_callback)

    transform_graph_widget = Dropdown(
        options=GRAPH_TRANSFORMATIONS,
        value=no_transformation,
        description='Transform'
    )

    transform_graph_widget, transform_graph_dynamic_params_container, transform_graph_dynamic_widgets_pool = generate_dynamic_widgets_pool(transform_graph_widget, GRAPH_TRANSFORMATIONS, TRANSFORMATION_STANDARD_ARGS, main_simulation_callback)


    # --- Attach Observers to Static Widgets ---
    static_widgets_to_observe = [
        graph_generator_widget,
        position_generator_widget,
        num_nodes_widget,
        show_edge_widget,
        transform_graph_widget
    ]

    def static_widget_observer(change):
        # print(f"Static widget changed: {change['owner'].description}") # Debug
        if main_simulation_callback:
            main_simulation_callback()

    for w in static_widgets_to_observe:
        w.observe(static_widget_observer, names='value')


    # --- Package Widgets and Layout ---
    widgets_dict = {
        'num_nodes': num_nodes_widget,
        'graph_generator': graph_generator_widget,
        'dynamic_widgets_pool': graph_generator_dynamic_widgets_pool, # Keep pool ref for getting values later
        'show_edge': show_edge_widget,
        'position_graph': position_generator_widget,
        'dynamic_widgets_pool_position': position_generator_dynamic_widgets_pool, # Keep pool ref for getting values later
        # Include the container in the dict if needed for layout reference outside
        'all_dynamic_graph_params_container': graph_generator_dynamic_params_container,
        'all_dynamic_position_params_container': position_generator_dynamic_params_container,
        'transform_graph': transform_graph_widget,
        'dynamic_widgets_pool_transform': transform_graph_dynamic_widgets_pool, # Keep pool ref for getting values later
        'all_dynamic_transform_params_container': transform_graph_dynamic_params_container
    }

    generator_box = VBox([
        graph_generator_widget,
        graph_generator_dynamic_params_container,
    ],layout=small_card_layout)

    transform_box = VBox([
        transform_graph_widget,
        transform_graph_dynamic_params_container,
    ],layout=small_card_layout)

    position_box = VBox([
        position_generator_widget,
        position_generator_dynamic_params_container,
        show_edge_widget,
    ],layout=small_card_layout)

    # Define the layout - place the container holding ALL dynamic widgets
    ui_layout = Box([
        Title,
        num_nodes_widget,
        generator_box,
        transform_box,
        position_box,
    ],layout=card_layout)

    return widgets_dict, ui_layout


# OPINION WIDGETS

OPINION_GENERATORS_UNSORTED = {
    'Zero': zero_opinion,
    'Random': random_opinion,
    'Identic': identic_opinion,
    'Symetric': symetric_opinion,
    'Symetric Random': symetric_random,
    'Manual': manual_opinion,
    'Eigendecomposition': eigendecomposition_opinion,
    'Eigendecomposition with Noise': eigendecomposition_opinion_with_noise,
}

OPINION_GENERATORS = {key: value for key, value in sorted(OPINION_GENERATORS_UNSORTED.items(),key=lambda x:x[0].lower())}

OPINION_STANDARD_ARGS = {'num_nodes', 'adj_matrix'}

def create_opinion_widgets(main_simulation_callback=None):
    """
    Creates opinion generator widgets, including pre-created dynamic parameter widgets
    which are hidden/shown based on opinion generator type selection.
    Uses the Manual Observation pattern for triggering plots.
    """

    # Make the opinion generator widget
    opinion_generator_widget = Dropdown(
        options=OPINION_GENERATORS,
        value=random_opinion,
        description='Generator'
    )

    opinion_generator_widget, opinion_generator_dynamic_params_container, opinion_generator_dynamic_widgets_pool = generate_dynamic_widgets_pool(opinion_generator_widget, OPINION_GENERATORS, OPINION_STANDARD_ARGS, main_simulation_callback)

    # --- Package Widgets and Layout ---
    widgets_dict = {
        'opinion_generator': opinion_generator_widget,
        'dynamic_widgets_pool': opinion_generator_dynamic_widgets_pool, # Keep pool ref for getting values later
        'all_dynamic_opinion_params_container': opinion_generator_dynamic_params_container
    }

    Title = Label(value='Opinion Generator', style={'font_weight': 'bold'})

    opinion_specific_box = VBox([
        opinion_generator_widget,
        opinion_generator_dynamic_params_container,
    ],layout=small_card_layout)

    # Define the layout - place the container holding ALL dynamic widgets
    ui_layout = Box([
        Title,
        opinion_specific_box,
    ],layout=card_layout)

    return widgets_dict, ui_layout
                        
# SIGNAL WIDGETS

SIGNAL_FUNCTIONS_UNSORTED ={
    'Sign': signal_sign,
    'Identity': signal_identity,
    'max(-1, min(Kx, 1))': signal_k_linear,
    'Two linear sign': signal_two_linear,
    'Tanh': signal_tanh,
    'Arctan': signal_arctan,
    'Upsin': signal_upsin,
    'With Deadzone Tanh': signal_with_deadzone_tanh,
    'Deadzone': signal_deadzone,
    'Discontinuity Tanh': signal_discontinuity_tanh,
    'Sinus': signal_sinus,
    'Step': step_func,
    'Smooth Step': smooth_step_func,
    'F_nm': F_nm_scalar,
}

SIGNAL_FUNCTIONS = {key: value for key, value in sorted(SIGNAL_FUNCTIONS_UNSORTED.items(),key=lambda x:x[0].lower())}

# --- Dummy definitions for stand-alone execution ---
SIGNAL_STANDARD_ARGS = {'x', 'x0', 'y0'}



# def wrap_signal(signal, x0, scalling):
    # accept AND forward any keyword‐args to the actual signal()
#     return lambda x, **kwargs: signal(x + x0, **kwargs) * scalling


def create_signal_widgets(main_simulation_callback=None):
    """
    Creates signal widgets, including pre-created dynamic parameter widgets
    which are hidden/shown based on signal type selection.
    Uses the Manual Observation pattern for triggering plots.
    """

    Title = Label(value='Opinion Signal Generator', style={'font_weight': 'bold'})

    # --- Static Widgets ---
    x0_widget = FloatSlider(value=0.0, min=-1.0, max=1.0, description=r'x0', continuous_update=False)
    y0_widget = FloatSlider(value=0.0, min=-1.0, max=1.0, description=r'y0', continuous_update=False)
    scalling_widget = FloatSlider(value=1.0, min=-1.0, max=1.0, description=r'Scaling', continuous_update=False)

    # Make the graph generator widget
    signal_widget = Dropdown(
        options=SIGNAL_FUNCTIONS,
        value=signal_identity,
        description='Signal'
    )

    plot_signal_output = Output()

    # --- Define the plot update callback BEFORE passing it ---
    @plot_signal_output.capture(clear_output=True, wait=True)
    def update_plot_signal(change=None):


        fig = plt.figure(figsize=(2, 2)) # Increased size slightly for better visibility
        ax = fig.add_subplot(111) # Add subplot

        # Get the selected function and its dynamic arguments
        selected_func_name = None
        try:
            # Find the name corresponding to the selected value
            selected_func_name = list(signal_widget.options.keys())[
                list(signal_widget.options.values()).index(signal_widget.value)
            ]
        except ValueError:
             print(f"Error: Selected signal value not found in options: {signal_widget.value}")
             plt.text(0.5, 0.5, "Error: Invalid signal selected", horizontalalignment='center', verticalalignment='center')
             plt.show()
             return # Exit if signal value is invalid

        dynamic_signal_args = {}
        # Check if dynamic_generator_widgets_pool is available and the selected function has dynamic widgets
        # This pool is created by generate_dynamic_widgets_pool, so it might not be available yet on first run?
        # No, generate_dynamic_widgets_pool is called below, so it will be available by the time
        # update_plot_signal is called by the initial update_plot_signal() call or observers.
        if 'signal_generator_dynamic_widgets_pool' in locals() and selected_func_name in signal_generator_dynamic_widgets_pool:
            for param_name, widget in signal_generator_dynamic_widgets_pool[selected_func_name].items():
                if hasattr(widget, 'value'):
                    # Ensure type consistency, especially for text inputs
                    value = widget.value
                    if isinstance(widget, (FloatSlider, FloatText)):
                        try: value = float(value)
                        except ValueError: print(f"Warning: Could not convert {param_name} value '{value}' to float.")
                    elif isinstance(widget, (IntSlider, IntText)):
                        try: value = int(value)
                        except ValueError: print(f"Warning: Could not convert {param_name} value '{value}' to int.")
                    # Handle Checkbox (bool) or Dropdown (any type)
                    dynamic_signal_args[param_name] = value
                else:
                     print(f"Warning: Dynamic widget for '{param_name}' has no 'value' attribute.")

        signal = signal_widget.value # This is the callable function

        # Prepare arguments for make_signal_jit
        x = np.linspace(-1, 1, 100)

        # Get expected dynamic parameter names from the signal function's signature
        # Exclude standard args ('x', 'x0', 'scalling')
        try:
            sig = inspect.signature(signal)
            expected_param_names = [p.name for p in sig.parameters.values() if p.name not in SIGNAL_STANDARD_ARGS]

            # Construct dynamic_args_tuple in the correct order expected by the signal function
            dynamic_tuple = tuple(dynamic_signal_args.get(name) for name in expected_param_names)

            # Check if all required dynamic arguments were collected
            missing_args = [name for name in expected_param_names if name not in dynamic_signal_args]
            if missing_args:
                 print(f"Warning: Missing dynamic arguments for '{selected_func_name}': {missing_args}")
                 # Potentially handle this case - maybe use defaults or skip plotting?
                 # For now, the call to make_signal_jit might fail if the signal function requires these.
                 # A robust solution would check sig.parameters and their defaults/requirements.
                 # For this fix, we assume _create_widget_for_param ensures widgets exist for required params.

            jit_signal = make_signal_jit(signal, x0=x0_widget.value, y0=y0_widget.value, scalling=scalling_widget.value, dynamic_args_tuple=dynamic_tuple)
            wrapped_signal = make_signal_vector(jit_signal) # Apply vectorization if needed

            # Plot the signal and the identity line
            try:
                ax.plot(x, wrapped_signal(x), label=selected_func_name)
                ax.plot(x, x, 'r--', label='Identity (x)')
                # ax.set_xlabel('Input x')
                # ax.set_ylabel('Signal Output')
                # ax.set_title(f'{selected_func_name} Signal')
                # ax.legend()
                ax.grid(True)
                # ax.set_ylim([-1.2, 1.2]) # Keep limits consistent
            except Exception as plot_err:
                 print(f"Error during plotting: {plot_err}")
                 ax.text(0.5, 0.5, f"Error plotting: {plot_err}", horizontalalignment='center', verticalalignment='center')


        except Exception as sig_err:
             print(f"Error processing signal '{selected_func_name}': {sig_err}")
             ax.text(0.5, 0.5, f"Error processing signal: {sig_err}", horizontalalignment='center', verticalalignment='center')

        plt.show()
        # print("Signal plot updated.") # Debug print

    # plot_signal_function_callback = None

    signal_widget, signal_generator_dynamic_params_container, signal_generator_dynamic_widgets_pool = generate_dynamic_widgets_pool(signal_widget, SIGNAL_FUNCTIONS, SIGNAL_STANDARD_ARGS, main_simulation_callback, other_callback=update_plot_signal)


    # plot_signal_output = Output()

    # @plot_signal_output.capture(clear_output=True, wait=True)
    # # when the signal function is changed, update the plot_signal_output
    # def update_plot_signal(change=None):
    #     fig = plt.figure(figsize=(2, 2))
    #     selected_func_name = list(signal_widget.options)[
    #         list(signal_widget.options.values()).index(signal_widget.value)
    #     ]
    #     dynamic_signal_args = {}
    #     if selected_func_name in signal_generator_dynamic_widgets_pool:
    #         for param_name, widget in signal_generator_dynamic_widgets_pool[selected_func_name].items():
    #             if hasattr(widget, 'value'):
    #                 value = widget.value
    #                 if isinstance(widget, (FloatSlider, FloatText)): value = float(value)
    #                 elif isinstance(widget, (IntSlider, IntText)):   value = int(value)
    #                 dynamic_signal_args[param_name] = value
    #     signal = signal_widget.value
    #     x = np.linspace(-1,1,100)
    #     sig = inspect.signature(signal)
    #     param_names = [p.name for p in sig.parameters.values() if p.name not in ('x',)]
    #     dynamic_tuple = tuple(dynamic_signal_args[name] for name in param_names)
    #     jit_signal = make_signal_jit(signal, x0=x0_widget.value, scalling= scalling_widget.value, dynamic_args_tuple=dynamic_tuple)
    #     wrapped_signal = make_signal_vector(jit_signal)
    #     plt.plot(x, wrapped_signal(x))
    #     plt.plot(x, x, 'r--')
    #     plt.show()

    # --- Attach Observers to Static Widgets ---
    static_widgets_to_observe = [
        x0_widget, y0_widget, scalling_widget, signal_widget 
    ]
    def static_widget_observer(change):
        # print(f"Static widget changed: {change['owner'].description}") # Debug
        if plot_signal_function_callback:
            plot_signal_function_callback()
        if main_simulation_callback:
            main_simulation_callback()


    for w in static_widgets_to_observe:
        w.observe(static_widget_observer, names='value')

        


    # --- Package Widgets and Layout ---
    widgets_dict = {
        'x0': x0_widget,
        'y0': y0_widget,
        'scalling': scalling_widget,
        'signal_generator': signal_widget,
        'dynamic_widgets_pool': signal_generator_dynamic_widgets_pool, # Keep pool ref for getting values later
        # Include the container in the dict if needed for layout reference outside
        'all_dynamic_signal_params_container': signal_generator_dynamic_params_container
    }

    signal_specific_box = VBox([
        signal_widget,
        signal_generator_dynamic_params_container,
    ],layout=small_card_layout)

    wrapped_signal_specific_box = VBox([
        x0_widget,
        y0_widget,
        scalling_widget,
    ],layout=small_card_layout)

    # Define the layout - place the container holding ALL dynamic widgets
    ui_layout = Box([
        Title,
        signal_specific_box,
        wrapped_signal_specific_box,
        plot_signal_output
    ],layout=card_layout)
      
    # ── Here's the one crucial line ──
    # wire up the global callback so all observers actually call update_plot_signal:
    plot_signal_function_callback = update_plot_signal

    update_plot_signal()
    
    return widgets_dict, ui_layout

def create_simulation_widgets(main_simulation_callback=None):
    time_step_widget = BoundedFloatText(value=0.1, min=0.001, max=1.0, step=0.001, description='Time Step', readout_format='.3f', continuous_update=False)
    discretization_method_widget = Dropdown(options=['euler', 'rk4'], value='euler', description='Discretization')
    T_widget = BoundedIntText(value=10, min=1, max=1000, step=1, description='Total Time', continuous_update=False)

    static_widgets_to_observe = [
        time_step_widget,
        discretization_method_widget,
        T_widget,
    ]
    def simulation_widget_observer(change):
        # print(f"Static sim/env/plot widget changed: {change['owner'].description}") # Debug
        if main_simulation_callback:
            main_simulation_callback()

    for w in static_widgets_to_observe:
        if hasattr(w, 'value'): w.observe(simulation_widget_observer, names='value')

    widgets_dict = {
        'time_step': time_step_widget,
        'discretization_method': discretization_method_widget,
        'T': T_widget,
    }

    ui_layout = Box([
        Label(value='Simulation Settings', style={'font_weight': 'bold'}),
        time_step_widget, discretization_method_widget, T_widget,
    ],layout=card_layout)

    return widgets_dict, ui_layout



def create_all_widgets(main_simulation_callback=None):
    graph_w, graph_ui = create_graph_widgets(main_simulation_callback=main_simulation_callback)
    opinion_w, opinion_ui = create_opinion_widgets(main_simulation_callback=main_simulation_callback)
    signal_w, signal_ui = create_signal_widgets(main_simulation_callback=main_simulation_callback)
    simulation_w, simulation_ui = create_simulation_widgets(main_simulation_callback=main_simulation_callback)
    
    full_widgets_dict = {}
    full_widgets_dict.update(graph_w)
    full_widgets_dict.update(opinion_w)
    full_widgets_dict.update(signal_w)
    full_widgets_dict.update(simulation_w)

    full_widgets_dict['dynamic_widgets_pool'] = {
        'graph': graph_w['dynamic_widgets_pool'],
        'opinion': opinion_w['dynamic_widgets_pool'],
        'signal_opinion': signal_w['dynamic_widgets_pool'],
        'position_graph': graph_w['dynamic_widgets_pool_position'],
        'transform_graph': graph_w['dynamic_widgets_pool_transform'],
        # 'simulation': simulation_w['dynamic_widgets_pool'],
    }



    ui_layout = Box([
        opinion_ui,
        simulation_ui,
        graph_ui,
        signal_ui,
    ])#,layout=Layout(border='1px solid black',width='100%', flex_direction='row', flex_wrap='wrap'))

    return full_widgets_dict, ui_layout
    


def get_dynamic_params_from_pool(dynamic_widgets_pool, selected_func, selected_func_name):
    """
    Collects current values from dynamic widgets associated with the selected function
    from the provided pool.
    """
    dynamic_params = {}
    # print("dynamic_widgets_pool", dynamic_widgets_pool)
    if not callable(selected_func):
        return dynamic_params # Cannot get params if no function is selected

    if selected_func_name in dynamic_widgets_pool:
        func_widgets_map = dynamic_widgets_pool[selected_func_name]
        # print(selected_func_name)
        for param_name, widget in func_widgets_map.items():
            # Only collect value if widget has one (e.g., exclude Labels)
            if hasattr(widget, 'value'):
                # Basic type guessing based on widget type
                value = widget.value
                if isinstance(widget, (FloatSlider, FloatText)):
                    value = float(value)
                elif isinstance(widget, (IntSlider, IntText)):
                    value = int(value)
                # Text widgets could be floats or ints, more sophisticated parsing needed if critical
                elif isinstance(widget, Textarea): # Handle Textarea (e.g., manual opinion)
                     value = widget.value # Keep as string for parsing by generator function
                elif isinstance(widget, Text):
                     # Attempt float conversion first, then int, then keep as string
                     try: value = float(value)
                     except ValueError:
                          try: value = int(value)
                          except ValueError: pass # Keep as string
                # Checkbox value is already bool

                dynamic_params[param_name] = value
    # else:
    #     print(f"Warning: Dynamic widgets for '{selected_func_name}' not found in pool.") # Debug

    return dynamic_params

