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

def sis(state, beta, gamma):
    if state.shape[0] != 2:
        raise ValueError("State be 2D representing S and I populations.")
    
    x_dot = np.zeros(2)
    x_dot[0] = -(beta * state[0] * state[1]) + gamma * state[1]
    x_dot[1] = (beta * state[0] * state[1]) - gamma * state[1]
    return x_dot

def closed_form_sis_i(t, beta, gamma, i_0):
    R0 = beta / gamma
    numerator = 1 - 1 / R0
    denominator = 1 + ((1 - 1 / R0 - i_0) / i_0) * np.exp(-(beta - gamma) * t)
    return numerator / denominator
