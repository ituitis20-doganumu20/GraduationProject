import random
import os

import os
import random

# Parameters
num_files = 32  # Number of output files to create
bits_per_file = 1000000  # Number of bits per output file

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the output directory if it doesn't exist
output_dir = os.path.join(script_dir, 'native_rand_out')
os.makedirs(output_dir, exist_ok=True)

# Generate and save random bits for each output file
for i in range(num_files):
    # Generate a random binary string of desired length
    random_bits = [random.choice([0, 1]) for _ in range(bits_per_file)]
    binary_string = ''.join(map(str, random_bits))
    
    # Create a unique file name for each file
    output_file_path = os.path.join(output_dir, f'pyrand_out_{i+101}.txt')
    
    # Save the binary string to the file
    with open(output_file_path, 'w') as output_file:
        output_file.write(binary_string)

    print("Random bits saved to:", output_file_path)

