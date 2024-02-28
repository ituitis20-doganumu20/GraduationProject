import os

# Assuming the binary file is in the same directory as your script
binary_file_path = 'output.bin'
text_file_name = 'output.txt'

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
binary_file_path = os.path.join(script_dir, binary_file_path)
text_file_path = os.path.join(script_dir, text_file_name)

# Open the binary file in binary read mode
with open(binary_file_path, 'rb') as binary_file:
    # Read the binary data
    binary_data = binary_file.read()
    
    # Convert each byte to its binary representation and concatenate
    binary_string = ''.join(format(byte, '08b') for byte in binary_data)
    
    # Save the binary string to a text file in the same directory as the script
    with open(text_file_path, 'w') as text_file:
        text_file.write(binary_string)

print("Binary data saved as binary string to:", text_file_path)
