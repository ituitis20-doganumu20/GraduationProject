# compression_model.py
import os
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

script_dir = os.path.dirname(os.path.abspath(__file__))
comp_model_path = os.path.join(script_dir, 'comp_model_weights.h5')
output_path = os.path.join(script_dir, 'output.bin')

# Function to generate random binary sequence as input
def generate_random_sequence(sequence_length):
    return np.random.randint(0, 2, size=(sequence_length,))

# Function to generate 32-bit sequences as target values
def generate_bits(num_samples):
    min_value = -2**31
    max_value = 2**31 - 1

    # Generate num_samples distinct integers
    distinct_integers = random.sample(range(min_value, max_value + 1), num_samples)

    # Convert integers to 32-bit binary representation
    binary_sequences = [format(num & 0xFFFFFFFF, '032b') for num in distinct_integers]

    # Convert binary sequences to lists of integers
    y_train = [[int(bit) for bit in sequence] for sequence in binary_sequences]

    return np.array(y_train)

# Function to create and train the compression model
def create_comp_model(input_size):
    # Create the compression model
    comp_model = Sequential([
        layers.Dense(1024, activation='relu', input_shape=(input_size,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='sigmoid')  
    ])

    # Compile the model
    comp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return comp_model

def train_comp_model(model, x_train):
    # Generate target values for training
    num_samples = len(x_train)
    y_train = generate_bits(num_samples)    
    # Train the compression model
    model.fit(x_train, y_train, epochs=100, batch_size=32)
    return model

def load_comp_model(input_size):
    try:
        # Load existing compression model
        model = load_model(comp_model_path)
        print("Compression model loaded successfully.")
    except (OSError, IOError):
        # Create a new compression model if loading fails
        print("Existing compression model not found. Creating a new model.")
        model = create_comp_model(input_size)  # Adjust initial input as needed
    return model

def save_comp_model(model):
    model.save(comp_model_path)
    print("Compression model saved successfully.")

def test_whole(initial_model, comp_model, input_size):
    # Get user input for the number of bits to be created
    num_bits_wanted = int(input("Enter the number of bits to be created: "))

    # Determine the number of iterations based on the input size (1024 bits)
    num_iterations = num_bits_wanted // input_size

    # Initialize an empty list to store the final output
    final_output = []

    # Initial random input for the first iteration
    input_test = np.array([generate_random_sequence(32)])
    
    # Loop through the iterations
    for _ in range(num_iterations):
        # Generate 1024-bit sequence using the initial model
        sequence = initial_model.predict(input_test)
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
    
    # Convert the binary predictions to a binary string
    binary_representation = ''.join(map(str, final_output))
    # Save the binary representation to prediction.bin
    with open(output_path, 'wb') as output_file:
        output_file.write(binary_representation.encode('utf-8'))

    # Print the file path for reference
    print("Binary Predictions saved to:", output_path)
