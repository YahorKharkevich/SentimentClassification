# To print the procent there must be a file oceny_test_rec.out in the same directory with the scrypt

num_of_words = 5000
n_steps = 500

import time
start = time.time()
import numpy as np
import re

# good_rec = input()
# bad_rec = input()
# test_rec = input()

good_rec = 'train_rec_pos'
bad_rec = 'train_rec_neg'
test_rec = 'test'

dct = {}
idx = 0
num_of_rec = 700

for num in range(0, num_of_rec):
    str_num = str(num).zfill(3)
    with open(f'{bad_rec}/train_n{str_num}', 'r', encoding='utf-8') as f:
        text = f.read()
    words = re.findall(r'[A-Za-z0-9]+', text.lower())
    s = set(words)
    for word in s:
        if (word not in dct):
            dct[word] = 1
        else:
            dct[word] += 1

for num in range(0, num_of_rec):
    str_num = str(num).zfill(3)
    with open(f'{good_rec}/train_p{str_num}', 'r', encoding='utf-8') as f:
        text = f.read()
    words = re.findall(r'[A-Za-z0-9]+', text.lower())
    s = set(words)
    for word in s:
        if (word not in dct):
            dct[word] = 1
        else:
            dct[word] += 1

dct = dict(sorted(dct.items(), key=lambda x: x[1], reverse=True)[:num_of_words])
size = len(dct)

for p in dct:
    dct[p] = idx
    idx += 1

vectors_x = []
vectors_y = []

for num in range(0, num_of_rec):
    str_num = str(num).zfill(3)
    with open(f'{bad_rec}/train_n{str_num}', 'r', encoding='utf-8') as f:
        text = f.read()
    words = re.findall(r'[A-Za-z0-9]+', text.lower())
    s = set(words)
    x = np.zeros(size)
    for word in s:
        if word in dct:
            x[dct[word]] += 1
    norm = np.linalg.norm(x)
    vectors_x.append(x / norm if norm > 0 else x)
    y = np.array(0)
    vectors_y.append(y)

for num in range(0, num_of_rec):
    str_num = str(num).zfill(3)
    with open(f'{good_rec}/train_p{str_num}', 'r', encoding='utf-8') as f:
        text = f.read()
    words = re.findall(r'[A-Za-z0-9]+', text.lower())
    s = set(words)
    x = np.zeros(size)
    for word in s:
        if word in dct:
            x[dct[word]] += 1
    norm = np.linalg.norm(x)
    vectors_x.append(x / norm if norm > 0 else x)
    y = np.array(1)
    vectors_y.append(y)

data_x = np.array(vectors_x)
data_y = np.array(vectors_y)

phis = np.zeros(size + 1)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def model(phis, x):
    pre = phis[0] + phis[1:] @ x
    return sigmoid(pre)


def compute_loss(data_x, data_y, phis):
    X = np.atleast_2d(data_x)
    y = np.squeeze(data_y).astype(float)
    b = phis[0]
    W = phis[1:]
    pre = b + X @ W
    pred = 1.0 / (1.0 + np.exp(-pre))
    loss = np.sum((pred - y) ** 2)
    return loss


def compute_gradient(data_x, data_y, phis):
    X = np.atleast_2d(data_x)
    y = np.squeeze(data_y).astype(float)
    b = phis[0]
    W = phis[1:]
    pre = b + X @ W
    pred = 1.0 / (1.0 + np.exp(-pre))
    diff = pred - y
    s = pred * (1.0 - pred)
    grad_b = 2.0 * np.sum(diff * s)
    grad_W = 2.0 * (X.T @ (diff * s))
    grad = np.concatenate(([grad_b], grad_W))
    return grad


def loss_function_1D(dist_prop, data_x, data_y, phi_start, search_direction):
    return compute_loss(data_x, data_y, phi_start + search_direction * dist_prop)


def line_search(data_x, data_y, phi, gradient, thresh=.001, max_dist=1.0, max_iter=20, verbose=False):
    a = 0
    b = 0.33 * max_dist
    c = 0.66 * max_dist
    d = 1.0 * max_dist
    n_iter = 0

    while np.abs(b - c) > thresh and n_iter < max_iter:
        n_iter = n_iter + 1
        lossa = loss_function_1D(a, data_x, data_y, phi, gradient)
        lossb = loss_function_1D(b, data_x, data_y, phi, gradient)
        lossc = loss_function_1D(c, data_x, data_y, phi, gradient)
        lossd = loss_function_1D(d, data_x, data_y, phi, gradient)
        if np.argmin((lossa, lossb, lossc, lossd)) == 0:
            b = a + (b - a) / 2
            c = a + (c - a) / 2
            d = a + (d - a) / 2
            continue
        if lossb < lossc:
            d = c
            b = a + (d - a) / 3
            c = a + 2 * (d - a) / 3
            continue
        a = b
        b = a + (d - a) / 3
        c = a + 2 * (d - a) / 3
    return (b + c) / 2.0


def gradient_descent_step(data_x, data_y, phi):
    gradient = compute_gradient(data_x, data_y, phi)
    direction = -gradient
    norm = np.linalg.norm(direction)
    unit_dir = direction / norm
    alpha = line_search(data_x, data_y, phi, unit_dir)
    phi = phi + unit_dir * alpha
    return phi

final_loss = 0

for c_step in range(n_steps):
    phis = gradient_descent_step(data_x, data_y, phis)
    loss = compute_loss(data_x, data_y, phis)
    final_loss = loss

ans = {}

with open(f'oceny_test_rec.out', 'r', encoding='utf-8') as f:
    text = f.read()
    d = dict(line.split() for line in text.splitlines())

errors = 0

for num in range(0, 600):
    str_num = str(num)
    with open(f'{test_rec}/rec_{str_num}', 'r', encoding='utf-8') as f:
        text = f.read()
    words = re.findall(r'[A-Za-z0-9]+', text.lower())
    s = set(words)
    x = np.zeros(size)
    for word in s:
        if word in dct:
            x[dct[word]] += 1
    norm = np.linalg.norm(x)
    x_norm = x / norm if norm > 0 else x
    if (model(phis, x_norm) >= 0.5):
        ans[str_num] = 1
    else:
        ans[str_num] = -1
    if (int)(d[f'rec_{str_num}']) != ans[str_num]:
        errors += 1
value = 100.0 - errors * 100 / 600
end = time.time()

print(f'The result is: {value:.2f}%')
print(f'Your loss is: {final_loss}')
print(f"Working time: {end - start:.2f} seconds")