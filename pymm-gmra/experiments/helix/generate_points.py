import numpy as np

def generate_helix_points(num_points):
    t = np.linspace(0, 4 * np.pi, num_points)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (4 * np.pi)
    return np.column_stack((x, y, z))

def write_points_to_file(points, filename='helix.txt'):
    with open(filename, 'w') as file:
        for point in points:
            file.write(f'{point[0]:.6f},{point[1]:.6f},{point[2]:.6f}\n')

if __name__ == '__main__':
    num_points = 10000  # Total number of points needed
    helix_points = generate_helix_points(num_points)
    write_points_to_file(helix_points)