import os
import tensorflow as tf
#from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import random
import numpy as np
import struct


script_dir = os.path.dirname(os.path.abspath(__file__))
comp_model_path = os.path.join(script_dir, 'bigger_comp_model_weights.h5')
model_path = os.path.join(script_dir, 'bigger_model_weights.h5')
#output_path = os.path.join(script_dir, 'output.bin')

input_size=1024

model = load_model(model_path)

comp_model = load_model(comp_model_path)

# Function to generate random binary sequence as input
def generate_random_sequence(sequence_length):
    return np.random.randint(0, 2, size=(sequence_length,))

# Get user input for the number of bits to be created
num_bits_wanted = int(input("Enter the number of bits to be created: "))

# Determine the number of iterations based on the input size (1024 bits)
num_iterations = num_bits_wanted // input_size

# Initialize an empty list to store the final output
final_output = []

# Initial random input for the first iteration
input_test = np.array([generate_random_sequence(32)])

loop=num_iterations
# Loop through the iterations
for _ in range(num_iterations):
    print(loop)
    loop-=1
    # Generate 1024-bit sequence using the initial model
    sequence = model.predict(input_test)
    # Apply a threshold to convert values to 0 or 1
    threshold = 0.5
    sequence_1024_bits = (sequence > threshold).astype(int)
    
    sequence_1024_bits_list = [int(bit) for bit in sequence_1024_bits.flatten()]
    
    # Concatenate the sequence to the final output
    final_output.extend(sequence_1024_bits_list)
    
    ####################################################################################
    
    # Compress the sequence to 32 bits using the compression model
    compressed_sequence = comp_model.predict(sequence_1024_bits)
    
    compressed_sequence_bit = (compressed_sequence > threshold).astype(int)
    
    # Set the compressed sequence as the input for the next iteration
    input_test = compressed_sequence_bit

# Print the final output
#print("Final Output:")
#print(final_output)


"""
# Convert the binary predictions to a binary string
binary_representation = ''.join(map(str, final_output))
# Save the binary representation to prediction.bin
with open(output_path, 'wb') as output_file:
    packed_bytes = bytearray()
    
    for i in range(0, len(final_output), 8):
        byte = 0
        for j in range(8):
            if i + j < len(final_output):
                byte |= final_output[i + j] << (7 - j)
        packed_bytes.append(byte)

    output_file.write(packed_bytes)

# Print the file path for reference
print("Binary Predictions saved to:", output_path)

"""


# Convert the final_output list into a single string of 0s and 1s
final_output_string = ''.join(map(str, final_output))

# Specify the path to the output text file
output_text_path = os.path.join(script_dir, 'output.txt')

# Save the concatenated string to a text file
with open(output_text_path, 'w') as file:
    file.write(final_output_string)