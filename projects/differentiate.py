import numpy as np


def differentiate(u, dt):
    """
    Function for differentiating using centered difference scheme.

    Inputs:
    u: np.ndarray, mesh function
    dt: float, timestep

    Output:
    d: np.ndarray, derivative of u
    """

    n = len(u)  # number of time steps
    d = np.zeros(n)  # empty array to store derivatives

    # Amplification factor
    A = 1 / (2 * dt)

    # End points
    d[0] = (u[1] - u[0]) / dt
    d[-1] = (u[-1] - u[-2]) / dt

    for i in range(1, n - 1):
        d[i] = (u[i + 1] - u[i - 1]) * A

    return d


def differentiate_vector(u, dt):
    """
    Function for differentiating using centered difference scheme, vectorized.

    Inputs:
    u: np.ndarray, mesh function
    dt: float, timestep

    Output:
    d: np.ndarray, derivative of u
    """

    n = len(u)  # number of time steps
    d = np.zeros(n)  # empty array to store derivatives

    # Amplification factor
    A = 1 / (2 * dt)

    # End points
    d[0] = (u[1] - u[0]) / dt
    d[-1] = (u[-1] - u[-2]) / dt

    d[1:-1] = (u[2:] - u[0:-2]) * A

    return d


def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)


if __name__ == "__main__":
    test_differentiate()
