import numpy as np

def generate_line_points(start_point, end_point, num_points):
    # Ensure start_point and end_point are numpy arrays
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    
    # Generate equally spaced points along the line
    points = np.linspace(start_point, end_point, num_points)
    
    return points

def write_points_to_file(points, filename='line.txt'):
    with open(filename, 'w') as file:
        for point in points:
            file.write(f'{point[0]:.6f},{point[1]:.6f},{point[2]:.6f}\n')

if __name__ == '__main__':
    start_point = [0, 0, 0]
    end_point = [1, 1, 1]
    num_points = 10000  # Total number of points needed
    line_points = generate_line_points(start_point, end_point, num_points)
    write_points_to_file(line_points)
