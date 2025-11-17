# Functions in this file are copied/modified from
# https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/utils.py

import numpy as np

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

def uppersphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) * 1 / 2 * 3 / 4    # [0, 3*pi/4]
    phi = v * 2 * np.pi
    return [phi, theta]

def circle_view_sequence(n, num_samples, offset=(0, 0)):
    """
    pitch always 45 deg
    yaw circles around
    """
    offset_0 = np.random.rand() * 2 - 1 # [-1, 1]
    theta = np.pi / 4 + offset_0 * 1.5 * np.pi / 8    # pitch
    phi = np.pi / 4 + (n + offset[1]) / num_samples * 2 * np.pi # yaw
    return [phi, theta]

def orthogonal_view_sequence(n, num_samples, offset=(0, 0)):
    """
    pitch always 45 deg
    yaw circles around
    """
    theta = 0    # pitch
    phi = (n) / num_samples * 2 * np.pi # yaw
    return [phi, theta]
