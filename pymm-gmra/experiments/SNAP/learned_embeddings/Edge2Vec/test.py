import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available CUDA devices
    device_count = torch.cuda.device_count()

    print(f"Number of CUDA devices available: {device_count}")

    # List details about each CUDA device
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)  # Convert to MB
        device_compute = torch.cuda.get_device_capability(i)

        print(f"Device {i}: {device_name}")
        print(f"    Memory: {device_memory:.2f} MB")
        print(f"    Compute Capability: {device_compute}")

else:
    print("CUDA is not available on this system.")
