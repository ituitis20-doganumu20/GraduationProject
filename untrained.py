import os
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
random_weights_path = os.path.join(script_dir, 'random_weights.h5')
comp_random_weights_path = os.path.join(script_dir, 'comp_random_weights.h5')

input_size = 32
output_size = 1024
target_length = 1_000_000  # 1 million bits

# Function to generate random binary sequence as input
def generate_random_sequence(sequence_length):
    return np.random.randint(0, 2, size=(sequence_length,))

def create_model(input_size):
    # Build the DCNN model
    model = models.Sequential([
        layers.Reshape((input_size, 1), input_shape=(input_size,)),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

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

    # Compile the compression model
    comp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return comp_model

def assign_random_weights(model):
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            random_weights = np.random.normal(size=layer.get_weights()[0].shape)
            random_biases = np.random.normal(size=layer.get_weights()[1].shape)
            layer.set_weights([random_weights, random_biases])

def generate_sequence(model, comp_model, initial_input, target_length):
    current_input = initial_input
    generated_sequence = []

    while len(generated_sequence) < target_length:
        # Predict with the main model
        output_main = model.predict(current_input.reshape(1, -1))
        output_main_binary = (output_main > 0.5).astype(int).flatten()

        # Append the output to the generated sequence
        generated_sequence.extend(output_main_binary)

        if len(generated_sequence) >= target_length:
            break

        # Use the compression model on the output of the main model
        output_comp = comp_model.predict(output_main_binary.reshape(1, -1))
        output_comp_binary = (output_comp > 0.5).astype(int).flatten()

        # Append the output to the generated sequence
        generated_sequence.extend(output_comp_binary)

        # Update the current input
        current_input = output_comp_binary[:input_size]

    return generated_sequence[:target_length]

# Create the models
model = create_model(input_size)
comp_model = create_comp_model(output_size)

# Assign random weights to both models
assign_random_weights(model)
assign_random_weights(comp_model)

# Generate the initial input
initial_input = generate_random_sequence(input_size)

# Generate the sequence
generated_sequence = generate_sequence(model, comp_model, initial_input, target_length)

# Print the first 100 bits of the generated sequence
print("First 100 bits of the generated sequence:")
print(generated_sequence[:100])

# Optionally, save the generated sequence to a file
output_file_path = os.path.join(script_dir, "generated_sequence.txt")
with open(output_file_path, "w") as file:
    file.write("".join(map(str, generated_sequence)))

print(f"Generated sequence saved to: {output_file_path}")
