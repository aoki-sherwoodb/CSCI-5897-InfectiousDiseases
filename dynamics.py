import numpy as np

def forward_euler(f, params: dict, x0: np.ndarray, delta_t: float, n: int, transient: int = 0, t0: float = 0.0, output_file: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward Euler ODE solver.
    Params:
        f: the m-dimensional ODE system to be solved numerically.
        params: the fixed parameters of the ODE system.
        x0: the state of the system at the starting time t0.
        delta_t: the solver time step.
        n: the number of time steps of the system to compute (excluding transient).
        transient: the number of time steps to discard as transient.
        t0: the starting time.
        output_file: optional, the path to a text file in which to write each state of the solution as a line.
    Returns:
        an m-by-n array consisting of n state vectors of the system solved forward in time and an array of the corresponding time steps.
    """
    n = n + transient
    m = x0.shape[0]
    solution = np.zeros((m, n))
    solution[:, 0] = x0
    timesteps = np.linspace(t0, t0 + (n - 1) * delta_t, n)

    for i in range(1, n):
        x_prev = solution[:, i-1]
        x_new = x_prev + delta_t * f(x_prev, **params)
        solution[:, i] = x_new
    
    if output_file is not None:
        with open(output_file, "w") as f:
            np.savetxt(f, solution, delimiter=",")

    return solution[:, transient:], timesteps[transient:]

def rk4(f, params: dict, x0: np.ndarray, delta_t: float, n: int, transient: int = 0, t0: float = 0.0, autonomous: bool = True, output_file: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Fourth-order Runge-Kutta ODE solver.
    Args:
        f: the m-dimensional ODE system to be solved numerically.
        params: the fixed parameters of the ODE system.
        x0: the state of the system at the starting time t0.
        delta_t: the solver time step.
        n: the number of time steps of the system to compute (excluding transient).
        transient: the number of time steps to discard as transient.
        t0: the starting time.
        autonomous: flag indicating whether the system is autonomous (not time-dependent).
        output_file: optional, the path to a text file in which to write each state of the solution as a line.
    Returns:
        an m-by-n array consisting of n state vectors of the system solved forward in time and an array of the corresponding time steps.
    """
    n = n + transient
    m = x0.shape[0]
    solution = np.zeros((m, n))
    solution[:, 0] = x0
    timesteps = np.linspace(t0, t0 + (n - 1) * delta_t, n)

    for i in range(1, n):
        x_prev = solution[:, i-1]
        delta_x = rk4_step(f, x_prev, t0 + (i - 1) * delta_t, delta_t, params, autonomous)
        solution[:, i] = x_prev + delta_x
    
    if output_file is not None:
        with open(output_file, "w") as f:
            np.savetxt(f, solution, delimiter=",")

    return solution[:, transient:], timesteps[transient:]

def rk4_adaptive(f, params: dict, x0: np.ndarray, n: int, transient: int = 0, t0: float = 0.0, tolerance: float = 0.001, autonomous: bool = True, output_file: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Adaptive timestep fourth-order Runge-Kutta ODE solver.
    Args:
        f: the m-dimensional ODE system to be solved numerically.
        params: the fixed parameters of the ODE system.
        x0: the state of the system at the starting time t0.
        n: the number of time steps of the system to compute (excluding transient).
        transient: the number of time steps to discard as transient.
        t0: the starting time.
        tolerance: the error tolerance for the adaptive timestep.
        autonomous: flag indicating whether the system is autonomous (not time-dependent).
        output_file: optional, the path to a text file in which to write each state of the solution as a line.
    Returns:
        an m-by-n array consisting of n state vectors of the system solved forward in time and an array of the corresponding time steps.
    """
    n = n + transient
    m = x0.shape[0]
    solution = np.zeros((m, n))
    solution[:, 0] = x0
    timesteps = np.zeros(n)
    
    #set arbitrary default time step
    delta_t = 0.1
    timesteps[0] = delta_t
    t = t0
    for i in range(1, n):
        x_prev = solution[:, i-1]

        # binary search for larger step size within tolerance
        while True:
            # compute new position with two current time steps
            delta_x_1 = rk4_step(f, x_prev, t, delta_t, params, autonomous)
            intermediate_x = x_prev + delta_x_1
            delta_x_2 = rk4_step(f, intermediate_x, t + delta_t, delta_t, params, autonomous)
            single_delta_x = delta_x_1 + delta_x_2

            # compute new position with double time step
            double_delta_x = rk4_step(f, x_prev, t, 2 * delta_t, params, autonomous)

            if np.max(double_delta_x - single_delta_x) < tolerance:
                delta_t *= 2
            else:
                delta_x = single_delta_x
                break

        # binary search for smaller step size within tolerance
        while True:
            # compute new position with current time step
            full_delta_x = rk4_step(f, x_prev, t, delta_t, params, autonomous)

            # compute new position with two half steps
            delta_x_1 = rk4_step(f, x_prev, t, delta_t / 2, params, autonomous)
            intermediate_x = x_prev + delta_x_1
            delta_x_2 = rk4_step(f, intermediate_x, t + delta_t / 2, delta_t / 2, params, autonomous)
            half_delta_x = delta_x_1 + delta_x_2

            if np.max(half_delta_x - full_delta_x) > tolerance:
                delta_t /= 2
            else:
                delta_x = full_delta_x
                break

        solution[:, i] = x_prev + delta_x
        timesteps[i] = delta_t

        t += delta_t
    
    if output_file is not None:
        with open(output_file, "w") as f:
            np.savetxt(f, solution, delimiter=",")

    return solution[:, transient:], timesteps[transient:]

def rk4_step(f, x_prev, t, delta_t, params, autonomous):
    if autonomous: 
        delta_x = rk4_step_auto(f, x_prev, delta_t, params)
    else:
        delta_x = rk4_step_non_auto(f, x_prev, delta_t, t, params) 

    return delta_x

def rk4_step_auto(f, x_prev, delta_t, params):
    k1 = f(x_prev, **params)
    k2 = f(x_prev + (delta_t / 2) * k1, **params)
    k3 = f(x_prev + (delta_t / 2) * k2, **params)
    k4 = f(x_prev + delta_t * k3, **params)
    return (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def rk4_step_non_auto(f, x_prev, delta_t, t_prev, params):
    k1 = f(x_prev, t_prev, **params)
    k2 = f(x_prev + (delta_t / 2) * k1, t_prev + (delta_t / 2), **params)
    k3 = f(x_prev + (delta_t / 2) * k2, t_prev + (delta_t / 2), **params)
    k4 = f(x_prev + delta_t * k3, t_prev + delta_t, **params)
    return (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

