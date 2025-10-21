import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

data = np.array([[0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90],
                 [0.67, 0.85, 1.05, 1.00, 1.40, 1.50, 1.30, 1.54, 1.55, 1.68, 1.73, 1.60]])

def model(phi, x):
    y_pred = phi[0] + phi[1] * x
    return y_pred

def compute_loss(data_x, data_y, model, phi):
    pred_y = model(phi, data_x)
    loss = np.sum((pred_y - data_y) ** 2)
    return loss

def compute_gradient(data_x, data_y, phi):
    dl_dphi0 = -2 * sum(data_y - phi[0] - phi[1] * data_x)
    dl_dphi1 = -2 * sum(data_x * (data_y - phi[0] - phi[1] * data_x))
    return np.array([[dl_dphi0], [dl_dphi1]])

def loss_function_1D(dist_prop, data, model, phi_start, search_direction):
    return compute_loss(data[0, :], data[1, :], model, phi_start + search_direction * dist_prop)


def line_search(data, model, phi, gradient, thresh=.00001, max_dist=0.1, max_iter=15, verbose=False):
    a = 0
    b = 0.33 * max_dist
    c = 0.66 * max_dist
    d = 1.0 * max_dist
    n_iter = 0

    while np.abs(b - c) > thresh and n_iter < max_iter:
        n_iter = n_iter + 1
        lossa = loss_function_1D(a, data, model, phi, gradient)
        lossb = loss_function_1D(b, data, model, phi, gradient)
        lossc = loss_function_1D(c, data, model, phi, gradient)
        lossd = loss_function_1D(d, data, model, phi, gradient)

        # Rule #1 If point A is less than points B, C, and D then halve distance from A to points B,C, and D
        if np.argmin((lossa, lossb, lossc, lossd)) == 0:
            b = a + (b - a) / 2
            c = a + (c - a) / 2
            d = a + (d - a) / 2
            continue;

        # Rule #2 If point b is less than point c then
        #                     point d becomes point c, and
        #                     point b becomes 1/3 between a and new d
        #                     point c becomes 2/3 between a and new d
        if lossb < lossc:
            d = c
            b = a + (d - a) / 3
            c = a + 2 * (d - a) / 3
            continue

        # Rule #2 If point c is less than point b then
        #                     point a becomes point b, and
        #                     point b becomes 1/3 between new a and d
        #                     point c becomes 2/3 between new a and d
        a = b
        b = a + (d - a) / 3
        c = a + 2 * (d - a) / 3

    # Return average of two middle points
    return (b + c) / 2.0


def gradient_descent_step(phi, data, model):
    gradient = compute_gradient(data[0, :], data[1, :], phi)
    alpha = line_search(data, model, phi, gradient)
    phi = phi - alpha * gradient
    return phi

n_steps = 100000
phi_all = np.zeros((2, n_steps + 1))
phi_all[0, 0] = 1.6
phi_all[1, 0] = -0.5
loss = compute_loss(data[0, :], data[1, :], model, phi_all[:, 0:1])

for c_step in range(n_steps):
    phi_all[:, c_step + 1:c_step + 2] = gradient_descent_step(phi_all[:, c_step:c_step + 1], data, model)
    loss = compute_loss(data[0, :], data[1, :], model, phi_all[:, c_step + 1:c_step + 2])


