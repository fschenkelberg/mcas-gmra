# imports
import numpy as np
import torch as pt

import matplotlib.pyplot as plt

array = np.load("/scratch/f006dg0/mcas-gmra/pymm-gmra/experiments/helix/helix_inverse.npy")

# Simple Helix Shape
def helix():
    # Load points from 'helix.txt'
    return np.loadtxt('./helix/helix.txt', delimiter=',')

X = helix() # Using 10,000 pts here imported from the helix.txt file
X = pt.from_numpy(X.astype(np.float32))


array = np.load("/scratch/f006dg0/mcas-gmra/pymm-gmra/experiments/helix/helix_inverse.npy")


# for x in range(0,14):
#     inverse = array[:,:,x].T

#     # Plotting the helix in 3D
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')

#     # Extracting x, y, z coordinates from the loaded data
#     xs = inverse[:, 0]
#     ys = inverse[:, 1]
#     zs = inverse[:, 2]

#     # Plotting the helix points
#     ax.scatter(xs, ys, zs, c='r', marker='o')

#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')

#     plt.title('Helix Plot')
#     plt.savefig(f'helix_plot_{x}.png')  # Save the plot as 'helix_plot.png'

# print(X)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Extracting x, y, z coordinates from the loaded data
xs = X[:, 0]
ys = X[:, 1]
zs = X[:, 2]

# Plotting the helix points
ax.scatter(xs, ys, zs, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.title('Helix Plot')
plt.savefig(f'orginal_plot_.png')  # Save the plot as 'helix_plot.png'