import numpy as np
import torch

from torch.autograd import grad
from assignment_three.src.neural_network import GenericNN
from assignment_three.src.numerical_solver import initial_condition

desired_dx = [1 / 10, 1 / 100]
desired_dt = [dx ** 2 / 2 for dx in desired_dx]
boundary_conditions = [0, 0]
space_min_max = [0, 1]
time_min_max = [0, 1]


def trial(points, network_output):
    x = points[:, 0]
    t = points[:, 1]

    return ((1 - t) * torch.sin(np.pi * x) + x * (1 - x) * t * network_output.view(-1)).view(-1, 1)


def cost(d_utrial, dd_utrial):
    x = xt_vals[:, 0]
    t = xt_vals[:, 1]

    # u_trial_dt = torch.ones_like(u_trial)
    # u_trial_dx_dx = torch.ones_like(u_trial, requires_grad=True)

    # grad(u_trial, x, grad_outputs=u_trial_dx_dx, allow_unused=True, retain_graph=True)
    # grad(u_trial_dx_dx, x, grad_outputs=u_trial_dx_dx, allow_unused=True, retain_graph=True)
    # grad(u_hat, t, grad_outputs=u_trial_dt, allow_unused=True, retain_graph=True)

    n = u_trial.shape[0]

    return torch.sum((d_utrial[:, 1] - dd_utrial[:, 0]) ** 2) / n


def compute_gradient_penalty(network_output, network_input):
    # ref: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
    '''

    :param input: state[index]
    :param network: actor or critic
    :return: gradient penalty
    '''
    input_ = torch.tensor(network_input).requires_grad_(True)
    mask = torch.ones_like(network_output)
    gradients = grad(network_output, input_, grad_outputs=mask,
                     retain_graph=True, create_graph=True,
                     allow_unused=True)[0]  # get tensor from tuple
    gradients = mask.view(-1, 1)

    return gradients


NUM_EPOCHS = 10000
for dx, dt in zip(desired_dx, desired_dt):
    NN = GenericNN(num_input_features=2, num_output_features=1, num_hidden_layers=3, num_hidden_features=20)
    optim = torch.optim.SGD(NN.parameters(), lr=1.0e-3)

    spatial_resolution = int(1 / dx)
    temporal_resolution = int(1 / dt)

    xt_vals = torch.from_numpy(np.random.uniform(0, 1, size=(spatial_resolution, 2))).requires_grad_(True)
    x = xt_vals[:, 0].requires_grad_(True)
    t = xt_vals[:, 1].requires_grad_(True)

    for i in range(NUM_EPOCHS):
        optim.zero_grad()
        u_hat = NN(xt_vals)
        u_trial = trial(xt_vals, u_hat)

        u_trial.backward(torch.ones_like(u_trial), retain_graph=True)  # does not work
        d_utrial = xt_vals.grad
        u_trial.backward(torch.ones_like(u_trial), retain_graph=True)  # does not work
        dd_utrial = xt_vals.grad

        print(compute_gradient_penalty(u_trial, xt_vals))

        loss = cost(d_utrial, dd_utrial)
        loss.backward()

        optim.step()
