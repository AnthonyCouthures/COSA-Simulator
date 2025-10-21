import numpy as np

from ipywidgets import (VBox, HBox, Layout, Dropdown, IntSlider, FloatSlider,
                        Checkbox, IntText, Textarea, Output, Text, FloatText,
                        Button, Label) # Added Label

from numba import njit, float64, int64



# Decorator definition (needed if used)
def specify_widget_params(**param_specs):
    def decorator(func):
        func._widget_specs = param_specs
        return func
    return decorator

def make_signal_jit(base_signal, x0, y0, scalling, dynamic_args_tuple):
    """
    Returns an njit-compiled scalar function: float64(float64)
    f(x) = base_signal(x - x0, *dynamic_args_tuple) * scaling
    """
    @njit(fastmath=True)
    def signal_scalar(x):
        return base_signal(x - x0, *dynamic_args_tuple) * scalling + y0
    return signal_scalar


def make_signal_vector(signal_scalar):
    """
    Returns a Python function that maps 1D numpy arrays -> 1D numpy arrays
    by calling the compiled scalar on each element.
    """
    def signal_vector(x_array):
        out = np.empty_like(x_array)
        for i in range(x_array.shape[0]):
            out[i] = signal_scalar(x_array[i])
        return out
    return signal_vector

# Signal Functions

@njit(float64(float64), fastmath=True)
def signal_identity(x : float):
    return x

@specify_widget_params(
        k_affine = {'widget': FloatSlider, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'default': 1.0, 'description': 'K'}
)
@njit(float64(float64, float64), fastmath=True)
def signal_k_linear(x, k_affine):
    y = k_affine * x
    # simple scalar branches
    if y > 1.0:
        return 1.0
    elif y < -1.0:
        return -1.0
    else:
        return y
    
@specify_widget_params(
        k_left = {'widget': FloatSlider, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'default': 1.0, 'description': 'K left'},
        k_right = {'widget': FloatSlider, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'default': 1.0, 'description': 'K right'},
        center = {'widget': FloatSlider, 'min': -1.0, 'max': 1.0, 'step': 0.1, 'default': 0.0, 'description': 'Center'},
)
@njit(float64(float64, float64, float64, float64), fastmath=True)
def signal_two_linear(x, k_left, k_right, center):
    if x < center:
        y = max(k_left * (x - center), -1.0)
    else:
        y = min(k_right * (x - center), 1.0)
    return y

@specify_widget_params(
    k_tanh = {'widget': FloatSlider, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'default': 1.0, 'description': 'K'}
)
@njit(float64(float64, float64), fastmath=True)
def signal_tanh(x, k_tanh):
    return np.tanh(x * k_tanh)

@specify_widget_params(
    k_arctan = {'widget': FloatSlider, 'min': 0.0, 'max': 10.0, 'step': 0.1, 'default': 5.0, 'description': 'K'}
)
@njit(float64(float64, float64), fastmath=True)
def signal_arctan(x: float, k_arctan: float):
    return np.arctan(x * k_arctan) / (np.pi/2)


@specify_widget_params(
    k_upsin = {'widget': FloatSlider, 'min': 0.0, 'max': 10.0, 'step': 0.1, 'default': 5.0, 'description': 'K'}
)
@njit(float64(float64, float64), fastmath=True)
def signal_upsin(x: float, k_upsin: float):
    y = np.sin(x * k_upsin)/k_upsin
    y_ = np.sin(x * k_upsin + np.pi)/k_upsin
    if y < y_:
        return x - y
    else:
        return x - y_


@specify_widget_params(
    K = {'widget': FloatSlider, 'min': 0.0, 'max': 10.0, 'step': 0.1, 'default': 10.0, 'description': 'K'},
    x_0 = {'widget': FloatSlider, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.1, 'description': 'Center'}
)
@njit(float64(float64, float64, float64), fastmath=True)
def signal_with_deadzone_tanh(x, K, x_0):
    y = np.tanh(K* (x+x_0)) -x_0
    y_ = np.tanh(K* (x-x_0)) +x_0
    if x < -x_0:
        return y
    elif x > x_0:
        return y_
    else:
        return x

# @specify_widget_params(
#     k_arctan_edge = {'min': 0.0, 'max': 10.0, 'step': 0.1, 'default': 5.0}
# )
# def signal_arctan_edge(x: float, k_arctan_edge: float, **kwargs):
#     return np.where(np.abs(x) < 1, np.arctan(x * k_arctan_edge / (x**2-1)) / (np.pi/2),  np.sign(x))

# @specify_widget_params(
#     delta_piecewise_sign = {'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.05}
# )
# def signal_piecewise_sign(x: float, delta_piecewise_sign: float, **kwargs):
#     return np.where(np.abs(x) > delta_piecewise_sign ,  np.sign(x), x/delta_piecewise_sign) 

@specify_widget_params(
    delta_smooth_step = {'widget': FloatSlider, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.05, 'description': 'δ'}
)
@njit(float64(float64, float64), fastmath=True)
def signal_smooth_step(x, delta_smooth_step):
    if x > delta_smooth_step:
        return 0.5 + (x - delta_smooth_step) / (2 * delta_smooth_step)
    elif x < -delta_smooth_step:
        return -0.5 + (x + delta_smooth_step) / (2 * delta_smooth_step)
    else:
        return 0.0



@specify_widget_params(
    delta_deadzone = {'widget': FloatSlider, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.25, 'description': 'δ'}
)
@njit(float64(float64, float64), fastmath=True)
def signal_deadzone(x, delta_deadzone):
    return x if abs(x) > delta_deadzone else 0.0

@specify_widget_params(
    k_discontinuity_tanh = {'widget': FloatSlider, 'min': 0.0, 'max': 10.0, 'step': 0.1, 'default': 1.0, 'description': 'K'}
)
@njit(float64(float64, float64), fastmath=True)
def signal_discontinuity_tanh(x, k_discontinuity_tanh):
    y = np.floor(abs(x) * k_discontinuity_tanh)
    return np.tanh(x / k_discontinuity_tanh * y)

@njit(float64(float64), fastmath=True)
def signal_sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

@specify_widget_params(
    k_sinus = {'widget': FloatSlider, 'min': 0.1, 'max': 10.0, 'step': 0.1, 'default': 1.0, 'description': 'K'}
)
@njit(float64(float64, float64), fastmath=True)
def signal_sinus(x, k_sinus):
    if abs(x) < 1e-6:
        return 0.0
    return x * (1 + np.sin(k_sinus * x) / (k_sinus * x))

@specify_widget_params(
    n_step = {'widget': IntSlider, 'min': 2, 'max': 10, 'step': 1, 'default': 2, 'description': 'N steps'},
)
@njit(float64(float64, int64), fastmath=True)
def step_func(x, n_step):
    # scalar step
    y = (x + 1.0) * 0.5 * n_step
    i = int(y) if y >= 0 else 0
    if i >= n_step:
        i = n_step - 1
    return -1.0 + (i + 0.5) * (2.0 / n_step)

@specify_widget_params(
    n_smooth_step = {'widget': IntSlider, 'min': 2, 'max': 10, 'step': 1, 'default': 2, 'description': 'N steps'},
    delta_smooth_step = {'widget': FloatSlider, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.05, 'description': 'δ'}
)
@njit(float64(float64, int64, float64), fastmath=True)
def smooth_step_func(x, n_smooth_step, delta_smooth_step):
# replicate original multi-boundary smoothing
    # clip x
    x_clip = 1.0 if x > 1.0 else (-1.0 if x < -1.0 else x)
    w = 2.0 / n_smooth_step
    # mid value
    idx = int(np.floor((x_clip + 1.0) / w))
    idx = 0 if idx < 0 else (n_smooth_step-1 if idx >= n_smooth_step else idx)
    mid = -1.0 + (idx + 0.5) * w
    # transition width
    delta = delta_smooth_step if delta_smooth_step <= w/2 else w/2
    slope = (w / (2.0 * delta)) if delta > 0.0 else 0.0
    # check all boundaries
    for k in range(1, n_smooth_step):
        b = -1.0 + k * w
        if abs(x_clip - b) < delta:
            m_prev = -1.0 + (k - 0.5) * w
            return m_prev + slope * (x_clip - (b - delta))
    return mid

import math


@specify_widget_params(
    n_plateaux = {'widget': IntSlider, 'min': 2, 'max': 10, 'step': 1, 'default': 2, 'description': 'N plateaux'},
    m_steepness = {'widget': IntSlider, 'min': 2, 'max': 10, 'step': 1, 'default': 2, 'description': 'Steepness'},
)
@njit(float64(float64, int64, int64), fastmath=True)
def F_nm_scalar(x, n_plateaux, m_steepness):
    """
    Numba‑compiled scalar version of F_nm:
      - x: single float in [-1,1]
      - n: number of plateaux (integer)
      - m: steepness exponent (integer)
    """
    # integration grid parameters
    N  = 500
    dx = 2.0/(N-1)
    # map x to grid index
    if x <= -1.0:
        return -1.0
    if x >=  1.0:
        return  1.0
    # compute threshold index once
    kf = (x + 1.0)*(N-1)/2.0
    k  = int(math.floor(kf))
    # accumulate both C and Y in one loop
    C = 0.0
    Y = 0.0
    for i in range(N):
        t = -1.0 + i*dx
        v = (1.0 + np.cos(n_plateaux * np.pi * t))**m_steepness
        C += v
        # only add to Y while t ≤ x
        if i <= k:
            Y += v
    # finish trapezoid scaling
    C *= dx
    Y *= dx
    # affine‑normalize to [−1,1]
    return -1.0 + 2.0 * Y / C

