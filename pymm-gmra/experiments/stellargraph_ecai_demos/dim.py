import sys

def parse_args_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read the first line from the file
            first_line = file.readline().strip()
            
            # Split the line on spaces
            args_array = first_line.split()

            # Return the length of the array
            return len(args_array)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <args_file>")
    else:
        args_file_path = sys.argv[1]
        result = parse_args_file(args_file_path)
        if result is not None:
            print(f"Length of args array: {result}")
