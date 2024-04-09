import numpy as np
import matplotlib.pyplot as plt

# Load the data from the file
data = np.load('./line/results/mnist_inverse.npy')

# 3d Plot
# Iterate over each set of points
for set_index in range(data.shape[0]):
    set_data = data[set_index]

    # Create a new 3D figure for each set of points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate over each dimension (x, y, z)
    for dim_index in range(set_data.shape[1]):
        dim_data = set_data[:, dim_index]

        # Plot the data for the current dimension
        ax.plot(np.arange(len(dim_data)), dim_data, zs=dim_index, label=f'Dimension {dim_index + 1}')

    # Set labels and title for the 3D line plot
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_zlabel('Dimension')
    ax.set_title(f'Set {set_index + 1} - 3D Line Plot')
    # ax.legend()

    # Save the plot to a file
    plt.savefig(f'./line/plots/set_{set_index + 1}_3d_line_plot.png')

    # Close the plot to free memory (optional)
    plt.close()

"""
# 2d Plots
# Iterate over each set of points
for set_index in range(data.shape[0]):
    set_data = data[set_index]

    # Iterate over each dimension (x, y, z)
    for dim_index in range(set_data.shape[1]):
        dim_data = set_data[:, dim_index]

        # Plot the data for the current dimension
        plt.figure()
        plt.plot(dim_data)
        plt.xlabel('Index')
        plt.ylabel(f'Dimension {dim_index + 1} Value')
        plt.title(f'Set {set_index + 1}, Dimension {dim_index + 1}')
        plt.savefig(f'./line/plots/set_{set_index + 1}_dim_{dim_index + 1}_plot.png')

        # Close the plot to free memory (optional)
        plt.close()
"""

"""
# Orginal Plot
# Load the data from the text file
data = np.loadtxt('./line/line.txt', delimiter=',')

# Separate the data into x, y, and z components
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Plot the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')

# Set labels and title for the 3D scatter plot
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Line Original Plot')

# Save the 3D scatter plot to a file
plt.savefig('./line/plots/line_original_plot.png')
"""
