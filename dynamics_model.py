'''
params = {
    'm1': 1.0, 'm2': 1.0,
    'l1': 0.3, 'l2': 0.3,
    'I1': 0.01, 'I2': 0.01,
    'g': 9.81
}
x = np.zeros(8)
u = 1.0

xdot_symbolic = f_symbolic(x, u, params)
xdot_fd = f_finite_difference(x, u, params)
'''
import numpy as np
from numpy import sin, cos

# --- Symbolic dynamics components (generated via SymPy) ---

def M_func(q, m1, m2, l1, l2, I1, I2):
    x, z, theta, phi = q
    M = np.zeros((4, 4))
    M[0, 0] = m1 + m2
    M[1, 1] = m1 + m2
    M[2, 2] = I1 + I2 + 0.25*l1**2*m1 + l2**2*m2 + l2*l1*m2*cos(phi)
    M[3, 3] = I2 + 0.25*l2**2*m2
    M[2, 3] = I2 + 0.5*l2**2*m2 + 0.5*l2*l1*m2*cos(phi)
    M[3, 2] = M[2, 3]
    return M

def G_func(q, m1, m2, l1, l2, g):
    _, z, theta, phi = q
    return np.array([
        0,
        -(m1 + m2) * g,
        -0.5 * g * (l1 * m1 * sin(theta) - 2 * l2 * m2 * sin(theta + phi)),
        0.5 * g * l2 * m2 * sin(theta + phi)
    ])

def Cqd_func(q, qd, m1, m2, l1, l2, I1, I2):
    _, _, theta, phi = q
    _, _, thetad, phid = qd
    Cqd = np.zeros(4)
    Cqd[0] = 0
    Cqd[1] = 0
    Cqd[2] = -0.5 * thetad * (
        l1 * m1 * thetad * sin(theta) - l2 * m2 * (phid + thetad) * sin(theta + phi)
    ) + 0.5 * l2 * m2 * phid * (phid + thetad) * sin(theta + phi)
    Cqd[3] = 0
    return Cqd

# --- f(x, u) using symbolic model ---

def f_symbolic(x, u, params):
    q = x[:4]
    qd = x[4:]

    M = M_func(q, params['m1'], params['m2'], params['l1'], params['l2'], params['I1'], params['I2'])
    G = G_func(q, params['m1'], params['m2'], params['l1'], params['l2'], params['g'])
    Cqd = Cqd_func(q, qd, params['m1'], params['m2'], params['l1'], params['l2'], params['I1'], params['I2'])

    tau = np.array([0.0, 0.0, 0.0, u])
    qdd = np.linalg.solve(M, tau - Cqd - G)

    return np.concatenate([qd, qdd])

# --- f(x, u) using finite differences (recursive) ---

def f_finite_difference(x, u, params, eps=1e-5):
    q = x[:4]
    qd = x[4:]
    n = len(q)

    # Approximate M using central differences
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            qd_plus = np.copy(qd)
            qd_minus = np.copy(qd)
            qd_plus[j] += eps
            qd_minus[j] -= eps

            x_plus = np.concatenate([q, qd_plus])
            x_minus = np.concatenate([q, qd_minus])

            f_plus = f_symbolic(x_plus, 0.0, params)[4 + i]
            f_minus = f_symbolic(x_minus, 0.0, params)[4 + i]

            M[i, j] = (f_plus - f_minus) / (2 * eps)

    # Approximate G
    G = np.zeros(n)
    for i in range(n):
        dq_vec = np.zeros(n)
        dq_vec[i] = eps
        x_pos = np.concatenate([q + dq_vec, qd])
        x_neg = np.concatenate([q - dq_vec, qd])
        G[i] = (f_symbolic(x_pos, 0, params)[4 + i] - f_symbolic(x_neg, 0, params)[4 + i]) / (2 * eps)

    # Approximate Cqd
    Cqd = np.zeros(n)
    for i in range(n):
        dq_vec = np.zeros(n)
        dq_vec[i] = eps
        x_pos = np.concatenate([q, qd + dq_vec])
        x_neg = np.concatenate([q, qd - dq_vec])
        Cqd[i] = (f_symbolic(x_pos, 0, params)[4 + i] - f_symbolic(x_neg, 0, params)[4 + i]) / (2 * eps)

    tau_vec = np.array([0.0, 0.0, 0.0, u])
    qdd = np.linalg.solve(M, tau_vec - Cqd - G)
    return np.concatenate([qd, qdd])
