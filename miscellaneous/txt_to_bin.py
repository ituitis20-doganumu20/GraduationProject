def txt_to_dat(input_txt_file, output_dat_file):
    try:
        # Read the input text file
        with open(input_txt_file, 'r') as txt_file:
            binary_string = txt_file.read().strip()
        
        # Ensure the string contains only 1s and 0s
        if not all(char in '01' for char in binary_string):
            raise ValueError("The input file should contain only 1s and 0s.")
        
        # Convert the string of 1s and 0s to bytes
        num_bytes = int(len(binary_string) / 8)
        binary_data = int(binary_string, 2).to_bytes(num_bytes, byteorder='big')
        
        # Write to the output .dat file
        with open(output_dat_file, 'wb') as dat_file:
            dat_file.write(binary_data)
        
        print(f"Successfully converted {input_txt_file} to {output_dat_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_txt_file = 'generated_sequence.txt'  # Replace with your input text file path
output_dat_file = 'output_untrained.dat'  # Replace with your desired output .dat file path
txt_to_dat(input_txt_file, output_dat_file)
