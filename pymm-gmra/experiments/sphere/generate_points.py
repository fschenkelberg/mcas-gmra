import numpy as np

def generate_sphere_points(total_points):
    num_points = int(np.sqrt(total_points))
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    return points

def write_points_to_file(points, filename='sphere.txt'):
    with open(filename, 'w') as file:
        for point in points:
            file.write(f'{point[0]:.6f},{point[1]:.6f},{point[2]:.6f}\n')

if __name__ == '__main__':
    total_points = 10000  # Total number of points needed
    sphere_points = generate_sphere_points(total_points)
    write_points_to_file(sphere_points)