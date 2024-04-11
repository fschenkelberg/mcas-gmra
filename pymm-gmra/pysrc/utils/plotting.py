import numpy as np
import matplotlib.pyplot as plt
import os 
# Load the data from the file
def plot_3d(filename, directory):
    data = np.load(filename)
    # 3d Plot
    # Iterate over each set of points
    for set_index in range(data.shape[2]):
        set_data = data[:, :, set_index]
        # Create a new 3D figure for each set of points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Iterate over each dimension (x, y, z)
        
        ax.scatter(set_data[0], set_data[1], set_data[2])
        # Set labels and title for the 3D line plot

        ax.set_title(f'Scale {set_index + 1} - 3D Plot')
        # ax.legend()
        # Save the plot to a file
        os.makedirs(f"{directory}/plots", exist_ok=True)
        plt.savefig(f'{directory}/plots/set_{set_index + 1}_3d_plot.png')
        # Close the plot to free memory (optional)
        plt.close()
        print(f"saved to {directory}/plots/set_{set_index + 1}_3d_plot.png")

# 2d Plots
#TODO: fix?
def plot_2d(filename):
    data = np.load(filename)
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
            os.makedirs("plots", exist_ok=True)
            plt.savefig(f'plots/set_{set_index + 1}_dim_{dim_index + 1}_plot.png')
            # Close the plot to free memory (optional)
            plt.close()


# Orginal Plot
def plot_data_from_txt(filename):
    data = np.load(filename)
    # Load the data from the text file
    data = np.loadtxt(filename, delimiter=',')
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
    os.makedirs("plots", exist_ok=True)
    plt.savefig('plots/line_original_plot.png')
