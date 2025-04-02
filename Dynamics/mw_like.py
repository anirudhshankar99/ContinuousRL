import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
G = 1  # Set G = 1 for simplicity

# Milky Way-like parameters
M_b, a_b = 1.0, 0.2  # Bulge parameters
M_d, a_d, b_d = 5.0, 3.0, 0.3  # Disk parameters
v_h, r_c = 0.8, 8.0  # Halo parameters

# Gravitational potentials
def phi_bulge(r):
    return -G * M_b / (r + a_b)

def phi_disk(R, z):
    return -G * M_d / np.sqrt(R**2 + (a_d + np.sqrt(z**2 + b_d**2))**2)

def phi_halo(r):
    return 0.5 * v_h**2 * np.log(r**2 + r_c**2)

# Full potential
def total_potential(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    R = np.sqrt(x**2 + y**2)
    return phi_bulge(r) + phi_disk(R, z) + phi_halo(r)

# Equations of motion (Hamiltonian formulation)
def equations(t, w):
    x, y, z, vx, vy, vz = w
    r = np.sqrt(x**2 + y**2 + z**2)
    R = np.sqrt(x**2 + y**2)

    # Compute forces
    dphi_b = -G * M_b / (r + a_b)**2
    dphi_h = -v_h**2 / (r**2 + r_c**2)
    dphi_d_R = -G * M_d * R / (R**2 + (a_d + np.sqrt(z**2 + b_d**2))**2)**(3/2)
    dphi_d_z = -G * M_d * (a_d + np.sqrt(z**2 + b_d**2)) * z / \
               (np.sqrt(z**2 + b_d**2) * (R**2 + (a_d + np.sqrt(z**2 + b_d**2))**2)**(3/2))

    # Accelerations
    ax = (dphi_b + dphi_h) * x / r + dphi_d_R * x / R
    ay = (dphi_b + dphi_h) * y / r + dphi_d_R * y / R
    az = (dphi_b + dphi_h) * z / r + dphi_d_z

    return [vx, vy, vz, ax, ay, az]