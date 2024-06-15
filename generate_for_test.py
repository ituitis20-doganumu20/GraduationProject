import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
comp_model_path = os.path.join(script_dir, 'comp_model_weights.h5')
model_path = os.path.join(script_dir, 'model_weights.h5')

# Subfolder to save output files
output_folder = os.path.join(script_dir, 'generated_outputs')

# Create the subfolder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to generate random binary sequence as input
def generate_random_sequence(sequence_length):
    return np.random.randint(0, 2, size=(sequence_length,))

# Load the models
model = load_model(model_path)
comp_model = load_model(comp_model_path)

# Ask user for the number of bits per output file
num_bits_per_output = int(input("Enter the number of bits per output file: "))

# Number of 1024-bit iterations needed to create the desired output
num_iterations_per_file = num_bits_per_output // 1024

# Number of output files to generate
num_output_files = 1000

# Generate 1000 different output files
for i in range(num_output_files):
    # Initialize an empty list to store the final output for each file
    final_output = []
    
    # Initial random input for the first iteration
    input_test = np.array([generate_random_sequence(32)])
    
    # Loop through iterations to create the desired output
    for _ in range(num_iterations_per_file):
        # Generate a 1024-bit sequence using the initial model
        sequence = model.predict(input_test)
        
        # Apply a threshold to convert values to 0 or 1
        threshold = 0.5
        sequence_1024_bits = (sequence > threshold).astype(int)
        
        # Convert the 1024-bit array to a list of integers
        sequence_1024_bits_list = [int(bit) for bit in sequence_1024_bits.flatten()]
        
        # Concatenate the sequence to the final output
        final_output.extend(sequence_1024_bits_list)
        
        # Compress the sequence to 32 bits using the compression model
        compressed_sequence = comp_model.predict(sequence_1024_bits)
        
        # Apply threshold to the compressed sequence and set it as the input for the next iteration
        input_test = (compressed_sequence > threshold).astype(int)
    
    # Convert the final_output list into a single string of 0s and 1s
    final_output_string = ''.join(map(str, final_output))
    
    # Specify the path to the output text file with a unique name
    output_text_path = os.path.join(output_folder, f'output_{i+102}.txt')
    
    # Save the concatenated string to a text file
    with open(output_text_path, 'w') as file:
        file.write(final_output_string)
