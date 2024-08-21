import numpy as np
import matplotlib.pyplot as plt


def mesh_function(f, t):
    """
    Applies a mesh function to a time array.

    Input:
    f: Python function
    t: np.ndarray

    Output:
    mesh: np.ndarray
    """
    mesh = np.zeros(len(t))
    for i in range(len(t)):
        mesh[i] = f(t[i])

    return mesh


def func(t):
    """
    Applies a mesh function to a time t.

    Input:
    t: float

    Output:
    f: float
    """
    if 0 <= t <= 3:
        f = np.exp(-t)
    elif 3 < t <= 4:
        f = np.exp(-3 * t)
    else:
        raise ValueError("t must be in the range 0, 4.")
    return f


n = 20  # number of time steps
del_t = 0.1  # time interval
tn = n * del_t  # end time
tn = np.linspace(0, tn, n + 1)  # array with all time steps

mesh_fun = mesh_function(func, tn)

plt.plot(tn, mesh_fun)
plt.title("Exercise 1.1, Define a mesh function and visualize it.")
plt.xlabel("Time"), plt.ylabel("f")
plt.grid()
plt.show()


def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
