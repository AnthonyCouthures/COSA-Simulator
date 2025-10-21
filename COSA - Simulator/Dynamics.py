import numpy as np

def orthogonalize(a, b):
    return a - (a @ b) * b

def gram_schmidt_and_norms(a):
    norms = np.empty(a.shape[1], dtype=a.dtype)
    for i in range(a.shape[1]):
        norms[i] = np.linalg.norm(a[:,i])
        a[:,i] = a[:,i] / norms[i]
        for j in range(i+1, a.shape[1]):
            a[:,j] = orthogonalize(a[:,j], a[:,i])
    return a, norms

def orthogonalize_along_dim(a, b):
    # a and b are K x N matrix 
    # I want a K matrix that is the dot product of a and b over the last dimension
    dot = np.einsum('ij,ij->i', a, b)
    return a - dot[:,None] * b

def gram_schmidt_and_norms_along_dim(a):
    # a is a K x N x N tensor
    norms = np.empty((a.shape[0], a.shape[-1]), dtype=a.dtype)
    # print('norms.shape :', norms.shape)
    for i in range(a.shape[-1]):
        norms[...,i] = np.linalg.norm(a[...,i], axis=1)
        # print('norms[:,i].shape :', norms[:,i].shape)
        # print('a[:,:,i].shape :', a[:,:,i].shape)
        # print(norms[:,i].repeat(a.shape[1],1).T)
        a[...,i] = a[...,i] / norms[...,i].repeat(a.shape[-1],1).T
        # ok jusqu'ici
        for j in range(i+1, a.shape[-1]):
            a[...,j] = orthogonalize_along_dim(a[...,j], a[...,i])
    return a, norms



from numba import njit, prange


def make_signal_jit(base_signal, x0, scalling, dynamic_args_tuple):
    """
    Returns an njit-compiled function signal(x: float) -> float that applies:
        base_signal(x - x0, *dynamic_args_tuple) * scaling
    dynamic_args_tuple must be a tuple of floats matching the base_signal signature after x.
    """
    # capture constants x0, scaling, dynamic_args_tuple
    @njit
    def signal(x):
        # call base_signal with explicit tuple unpacking
        return base_signal(x - x0, *dynamic_args_tuple) * scalling
    return signal



def make_step_functions(norm_adj, dt, signal_jit):
    n = norm_adj.shape[0]

    @njit
    def step(x, norm_adj=norm_adj, dt=dt, signal_jit=signal_jit):
        f_x = np.empty(n, np.float64)
        for i in prange(n):
            acc = 0.0
            for j in range(n):
                acc += norm_adj[i, j] * signal_jit(x[j])
            f_x[i] = acc - x[i]
        for i in prange(n):
            x[i] += dt * f_x[i]
        return x

    @njit
    def step_rk4(x, norm_adj=norm_adj, dt=dt, signal_jit=signal_jit):
        k1 = np.empty(n, np.float64)
        k2 = np.empty(n, np.float64)
        k3 = np.empty(n, np.float64)
        k4 = np.empty(n, np.float64)
        tmp = np.empty(n, np.float64)

        # k1
        for i in prange(n):
            acc = 0.0
            for j in range(n):
                acc += norm_adj[i, j] * signal_jit(x[j])
            k1[i] = acc - x[i]
        # k2
        for i in prange(n):
            tmp[i] = x[i] + 0.5 * dt * k1[i]
        for i in prange(n):
            acc = 0.0
            for j in range(n):
                acc += norm_adj[i, j] * signal_jit(tmp[j])
            k2[i] = acc - tmp[i]
        # k3
        for i in prange(n):
            tmp[i] = x[i] + 0.5 * dt * k2[i]
        for i in prange(n):
            acc = 0.0
            for j in range(n):
                acc += norm_adj[i, j] * signal_jit(tmp[j])
            k3[i] = acc - tmp[i]
        # k4
        for i in prange(n):
            tmp[i] = x[i] + dt * k3[i]
        for i in prange(n):
            acc = 0.0
            for j in range(n):
                acc += norm_adj[i, j] * signal_jit(tmp[j])
            k4[i] = acc - tmp[i]

        for i in prange(n):
            x[i] += dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
        return x

    return step, step_rk4

import scipy
from signal_function import make_signal_vector

class DynamicalSystem:
    """
    Python class to choose among compiled stepping functions.
    """
    def __init__(self, x0, norm_adj, dt, signal_jit, method='euler'):
        self.x = x0.astype(np.float64)
        self.norm_adj = norm_adj / np.sum(np.abs(norm_adj), axis=1, keepdims=True)
        self.norm_adj = self.norm_adj.astype(np.float64)
        self.dt = float(dt)
        self.signal_jit = signal_jit
        # create jitted steppers
        self.step, self.step_rk4 = make_step_functions(self.norm_adj, self.dt, signal_jit)
        # select
        self.method = method

    def __call__(self, x=None):
        if x is None:
            x = self.x
        if self.method == 'rk4':
            return self.step_rk4(x)
        elif self.method == 'euler':
            return self.step(x)
        elif self.method == 'scipy':
            return scipy.integrate.odeint(self.step, x, [0, self.dt])[1]
    def run(self, n):
        for _ in range(n):
            self()

    def run_and_return_all(self, n):
        traj = np.empty((n+1, self.x.shape[0]), dtype=np.float64)
        traj[0] = self.x.copy()
        for i in range(1, n+1):
            self()
            traj[i] = self.x.copy()
        return traj

    def run_and_return_all_scipy(self, n):
        # solve_ivp is not working with njit
        def step_scipy(t, x):
            signal = make_signal_vector(self.signal_jit)
            step = self.norm_adj @ signal(x) - x
            return step
        return scipy.integrate.solve_ivp(step_scipy, [0, self.dt*n], self.x, method='DOP853').y
