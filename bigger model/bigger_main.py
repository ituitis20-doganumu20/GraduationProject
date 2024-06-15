import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np
from bigger_compression_model import *


script_dir = os.path.dirname(os.path.abspath(__file__))
comp_model_path = os.path.join(script_dir, 'bigger_comp_model_weights.h5')
file_path = os.path.join(script_dir, 'train_data.txt')
model_path = os.path.join(script_dir, 'bigger_model_weights.h5')
#output_file_path = os.path.join(script_dir, 'binary2.bin')
# Generate random sequences for training
num_samples = 1000
input_size = 32
output_size = 1024

try:
    with open(file_path, 'rb') as file:
        # Read the first 100 bytes of the file
        file_content = file.read(128*num_samples)

        # Display the binary representation
        binary_representation = ''.join(format(byte, '08b') for byte in file_content)
        #print(f"The first 100 bytes in binary: {binary_representation}")
        

        """
        # Save the binary representation to binary.bin
        with open(output_file_path, 'wb') as output_file:
            output_file.write(binary_representation.encode('utf-8'))
        """
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

# Function to load the model
def load_the_model(input_size):
    try:
        # Load existing model
        model = load_model(model_path)
        print("Model loaded successfully.")
    except (OSError, IOError):
        # Create a new model if loading fails
        print("Existing model not found. Creating a new model.")
        model = create_model(input_size)

    return model

# Convert binary representation to a list of integers
binary_list = [int(bit) for bit in binary_representation]

sample_size = len(binary_list) // num_samples
# Create training data
X_train = np.array([generate_random_sequence(input_size) for _ in range(num_samples)])
# Split binary_list into 100 samples for y_train
y_train = np.array([binary_list[i*sample_size : (i+1)*sample_size] for i in range(num_samples)])

#print("Binary List:", binary_list)
#print(len(binary_list))

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
        layers.Dense(768, activation='relu'),
        layers.Dense(1024, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model):    
    model.fit(X_train, y_train, epochs=200, batch_size=32)
    return model

def save_model(model):
    model.save(model_path)
    print("Model saved successfully.")

def test_model(model):
    # Generate new random input for testing
    input_test = np.array([generate_random_sequence(input_size)])
    print("here is test input:")
    print(input_test)
    # Make predictions using the trained model
    output_prediction = model.predict(input_test)

    # Apply a threshold to convert values to 0 or 1
    threshold = 0.5
    output_prediction_binary = (output_prediction > threshold).astype(int)

    # Print the generated pseudorandom sequence
    print("Generated Pseudorandom Sequence:")
    print(output_prediction.flatten().astype(int))
    print(output_prediction.flatten().astype(int)[100:105])

    # Specify the file path in the script directory
    file_path = os.path.join(script_dir, "generated_sequence.txt")

    # Save the flattened generated pseudorandom sequence to the file without new lines
    with open(file_path, "w") as file:
        file.write("".join(map(str, output_prediction_binary.ravel())))

    # Print the file path for reference
    print("Generated sequence saved to:", file_path)

    # Specify the file path in the script directory
    output_file_path = os.path.join(script_dir, "prediction.bin")

    # Flatten the list if it's nested
    flat_output = [item for sublist in output_prediction_binary.tolist() for item in sublist]

    # Convert the binary predictions to a binary string
    binary_representation = ''.join(map(str, flat_output))

    # Save the binary representation to prediction.bin
    with open(output_file_path, 'wb') as output_file:
        output_file.write(binary_representation.encode('utf-8'))

    # Print the file path for reference
    print("Binary Predictions saved to:", output_file_path)

trained=False

"""
# Function to clear the menu lines
def clear_menu_lines():
    # ANSI escape code to move the cursor to the beginning of the previous line
    print("\033[F\033[K" * 6)
"""
    
thereismodel=False

def ask_which():
    print("0 - Generative Model")
    print("1 - Compression Model")       
    choice = input("which model?: ")
    return choice

while True:
    #clear_menu_lines() 
    
    print("\nMenu:")
    print("0 - Create Model")
    print("1 - Load Model")
    print("2 - Train Model")
    print("3 - Test Model")
    print("4 - Save Model")

    choice = input("Enter your choice (0-4, or 'q' to quit): ")

    if choice=='0':
        choice2 = ask_which()
        if choice2 == '0':      
            model=create_model(input_size)
        elif choice2 == '1':
            comp_model = create_comp_model(1024) 
        thereismodel=True
        print("Model created.") 
    elif choice == '1':
        choice2 = ask_which()
        if choice2 == '0':      
            model = load_the_model(input_size)
        elif choice2 == '1':
            comp_model = load_comp_model(1024)
        thereismodel=True
    elif choice == '2':
        if thereismodel:
            choice2 = ask_which()
            if choice2 == '0':      
                model = train_model(model)
            elif choice2 == '1':
                comp_model = train_comp_model(comp_model, y_train)
            trained=True
        else:
            print("Please make the model first.") 
    elif choice == '3':
        if thereismodel:
            choice2 = ask_which()
            if choice2 == '0':      
                test_model(model)
            elif choice2 == '1':
                test_whole(model,comp_model,1024)
        else:
            print("Please make the model first.")
    elif choice == '4':
        if thereismodel:
            choice2 = ask_which()
            if choice2 == '0':      
                save_model(model)
            elif choice2 == '1':
                save_comp_model(comp_model)
        else:
            print("Please make the model first.")       
    elif choice.lower() == 'q':
        break
    else:
        print("Invalid choice. Please enter a number between 1 and 4, or 'q' to quit.")