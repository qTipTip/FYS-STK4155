import numpy as np
import tqdm


def ftcs(space_resolution, time_resolution, space_min_max=[0, 1], time_max=1, boundary_conditions=[0, 0],
         initial_condition=None):
    """
    Solves the 1D-heat equation using a forward-in-time centered-in-space discretization scheme.

    :param space_resolution: number of points in spatial direction
    :param time_resolution: number of points in temporal direction
    :param space_min_max: the spatial boundary values
    :param time_max: the time to run the simulation for
    :param boundary_conditions: the boundary conditions at the space_min_max-values
    :param initial_conditions: the initial conditions at time = 0, callable.
    """

    if initial_condition is None:
        initial_condition: lambda x: 0

    dt = time_max / time_resolution
    u0, dx = np.linspace(*space_min_max, num=space_resolution, retstep=True)
    u = np.zeros((time_resolution, space_resolution))
    u[:, 0] = boundary_conditions[0]
    u[:, -1] = boundary_conditions[1]
    u[0] = initial_condition(u0)

    F = dt / dx ** 2
    print(F)
    for step in tqdm.trange(time_resolution - 1):
        for i in range(1, space_resolution - 1):
            u[step + 1, i] = u[step, i] + F * (u[step, i - 1] - 2 * u[step, i] + u[step, i + 1])

    return u


@np.vectorize
def initial_condition(x):
    return np.sin(np.pi * x)


@np.vectorize
def exact_solution(x, t):
    return np.sin(np.pi(x)) * np.exp(-np.pi ** 2 * t)


if __name__ == '__main__':

    desired_dx = [1 / 10, 1 / 20, 1 / 30, 1 / 40]

    for dx in desired_dx:
        dt = 1 / 2 * dx ** 2

        N = int(1 / dx)
        T = int(1 / dt)

        u = ftcs(N, T, initial_condition=initial_condition)
        import matplotlib.pyplot as plt

        for step in u:
            plt.title(f'N = {N} T = {T}')
            plt.plot(step, color='gray', alpha=0.7)
        plt.show()
