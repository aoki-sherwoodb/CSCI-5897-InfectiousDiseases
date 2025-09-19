import numpy as np

def sir_birth_death(state, beta, gamma, mu_birth, mu_death):
    if state.shape[0] != 3:
        raise ValueError("State be 3D representing S, I, and R populations.")
    
    x_dot = np.zeros(3)
    x_dot[0] = -(beta * state[0] * state[1]) / state.sum() + mu_birth * (state.sum()) - mu_death * state[0] 
    x_dot[1] = (beta * state[0] * state[1]) / state.sum() - gamma * state[1] - mu_death * state[1]
    x_dot[2] = gamma * state[1] - mu_death * state[2]
    return x_dot

def sir(state, beta, gamma):
    if state.shape[0] != 3:
        raise ValueError("State be 3D representing S, I, and R populations.")
    
    x_dot = np.zeros(3)
    x_dot[0] = -(beta * state[0] * state[1]) / state.sum()
    x_dot[1] = (beta * state[0] * state[1]) / state.sum() - gamma * state[1]
    x_dot[2] = gamma * state[1]
    return x_dot