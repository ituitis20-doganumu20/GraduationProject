"""
tests multiple files. saves the avarage p values of each test to a json file. 
"""

import os
import json
from statistics import mean
from collections import defaultdict
from tests.FrequencyTest import FrequencyTest
from tests.RunTest import RunTest
from tests.Matrix import Matrix
from tests.Spectral import SpectralTest
from tests.TemplateMatching import TemplateMatching
from tests.Universal import Universal
from tests.Complexity import ComplexityTest
from tests.Serial import Serial
from tests.ApproximateEntropy import ApproximateEntropy
from tests.CumulativeSum import CumulativeSums
from tests.RandomExcursions import RandomExcursions

# Folder path containing text files
folder_path = "native_rand_out"  # Change to your folder path
output_file = "test_results_for_native.json"  # File to save averages and file names

# Load existing data from the JSON file, initializing if not existing
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        data = json.load(f)
    processed_files = data.get("processed_files", [])
    failed_files = data.get("failed_files", [])
    test_averages = defaultdict(lambda: defaultdict(list), data.get("test_averages", {}))
else:
    processed_files = []
    failed_files = []
    test_averages = defaultdict(lambda: defaultdict(list))

try:
    print("working, ctrl+c to quit.")
    # Process each file to extract and validate test results
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        if file_name in processed_files or any(failure['file_name'] == file_name for failure in failed_files):
            #print(f"File {file_name} already processed or failed. Skipping.")
            continue  # Skip files that have already been processed

        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as file:
            # Read binary data from the text file
            data_list = [line.strip().rstrip() for line in file]
            binary_data = ''.join(data_list)

            # Define the tests to run on the binary data
            results = [
                ("Frequency Test", FrequencyTest.monobit_test(binary_data[:1000000])),
                ("Block Frequency Test", FrequencyTest.block_frequency(binary_data[:1000000])),
                ("Run Test", RunTest.run_test(binary_data[:1000000])),
                ("Longest Run of Ones", RunTest.longest_one_block_test(binary_data[:1000000])),
                ("Binary Matrix Rank Test", Matrix.binary_matrix_rank_text(binary_data[:1000000])),
                ("Spectral Test", SpectralTest.spectral_test(binary_data[:1000000])),
                ("Non-overlapping Template Matching", TemplateMatching.non_overlapping_test(binary_data[:1000000])),
                ("Overlapping Template Matching", TemplateMatching.overlapping_patterns(binary_data[:1000000])),
                ("Universal Statistical Test", Universal.statistical_test(binary_data[:1000000])),
                ("Linear Complexity Test", ComplexityTest.linear_complexity_test(binary_data[:1000000])),
                ("Serial Test", Serial.serial_test(binary_data[:1000000])),
                ("Approximate Entropy Test", ApproximateEntropy.approximate_entropy_test(binary_data[:1000000])),
                ("Cumulative Sums (Forward)", CumulativeSums.cumulative_sums_test(binary_data[:1000000], 0)),
                ("Cumulative Sums (Backward)", CumulativeSums.cumulative_sums_test(binary_data[:1000000], 1)),
                ("Random Excursions Test", RandomExcursions.random_excursions_test(binary_data[:1000000])),
                ("Random Excursions Variant Test", RandomExcursions.variant_test(binary_data[:1000000]))
            ]
            
            # Check if any test in the results failed
            failed_test_names = []  # List to store names of failed tests

            for test_name, test_result in results:
                if test_name in ["Random Excursions Test", "Random Excursions Variant Test"]:
                    # Check the boolean at the fifth position in each tuple
                    if any(not item[4] for item in test_result):
                        failed_test_names.append(test_name)
                elif test_name == "Serial Test":
                    # Check the boolean at the second position in each sub-tuple
                    if any(not sub_tuple[1] for sub_tuple in test_result):
                        failed_test_names.append(test_name)
                else:
                    # For other tests, check the boolean at the second position
                    if not test_result[1]:
                        failed_test_names.append(test_name)

            if failed_test_names:
                # If any test failed, add this file to the failed files list
                failed_files.append({"file_name": file_name, "failed_tests": failed_test_names})
                print(f"File {file_name} failed at least one test: {', '.join(failed_test_names)}")
                continue  # Skip adding results to test_averages if any test failed

            # Initialize or update test_averages with new results
            for test_name, test_result in results:
                if test_name not in test_averages:
                    test_averages[test_name] = defaultdict(list)

                if test_name in ["Random Excursions Test", "Random Excursions Variant Test"]:
                    # Collect p-values from all sub-tests
                    for sub_test in test_result:
                        excursion_key = f"{sub_test[0]}"
                        test_averages[test_name][excursion_key].append(sub_test[3])
                elif test_name == "Serial Test":
                    # Separate p-values for both sub-tests
                    test_averages[test_name]["Serial 1"].append(test_result[0][0])
                    test_averages[test_name]["Serial 2"].append(test_result[1][0])
                else:
                    # Use a general key for single p-value tests
                    test_averages[test_name]["Main"].append(test_result[0])

        # Add the processed file name to the list
        processed_files.append(file_name)

        # Calculate the updated averages for each test
        for test_name, sub_tests in test_averages.items():
            for sub_test_name, p_values in sub_tests.items():
                # Calculate new average p-value
                avg_p_value = mean(p_values)
                test_averages[test_name][sub_test_name] = [avg_p_value] 

        # Save the updated data to the JSON file after processing each file
        with open(output_file, "w") as f:
            json.dump(
            {"processed_files": processed_files, "failed_files": failed_files, "test_averages": test_averages}, 
            f, 
            indent=4
            )

        # Print that this file is done
        print(f"File {file_name} is done.")

except KeyboardInterrupt:
    # Output the average p-values for verification
    print("Average p-values:")
    for test_name, sub_tests in test_averages.items():
        if len(sub_tests) == 1:
            sub_test_name, avg_p_values = list(sub_tests.items())[0]
            avg_p_value = avg_p_values[0]  # Extract the first item from the list
            print(f"{test_name}: {avg_p_value:.6f}")
        else:
            print(f"{test_name}:")
            for sub_test_name, avg_p_values in sub_tests.items():
                avg_p_value = avg_p_values[0]  # Extract the first item from the list
                print(f"  {sub_test_name}: {avg_p_value:.6f}")
                
    """
    # List files with failed tests and the tests that caused the failure
    if failed_files:
    print("Files with failed tests:")
    for failure in failed_files:
        file_name = failure["file_name"]
        failed_test_names = failure["failed_tests"]
        print(f"{file_name}: Failed Tests - {', '.join(failed_test_names)}")
    """
    
    total_files = len(processed_files) + len(failed_files)   # Total number of processed files
    failed_file_count = len(failed_files)  # Number of files that failed    
    # Calculate the percentage of failed files
    failed_percentage = (failed_file_count / total_files) * 100 if total_files > 0 else 0
    
    # Output the percentage of failed files
    print(f"Percentage of failed files: {failed_percentage:.2f}%")

# Output the average p-values for verification
print("Average p-values:")
for test_name, sub_tests in test_averages.items():
    if len(sub_tests) == 1:
        sub_test_name, avg_p_values = list(sub_tests.items())[0]
        avg_p_value = avg_p_values[0]  # Extract the first item from the list
        print(f"{test_name}: {avg_p_value:.6f}")
    else:
        print(f"{test_name}:")
        for sub_test_name, avg_p_values in sub_tests.items():
            avg_p_value = avg_p_values[0]  # Extract the first item from the list
            print(f"  {sub_test_name}: {avg_p_value:.6f}")
            
"""
# List files with failed tests and the tests that caused the failure
if failed_files:
print("Files with failed tests:")
for failure in failed_files:
    file_name = failure["file_name"]
    failed_test_names = failure["failed_tests"]
    print(f"{file_name}: Failed Tests - {', '.join(failed_test_names)}")
"""

total_files = len(processed_files) + len(failed_files)   # Total number of processed files
failed_file_count = len(failed_files)  # Number of files that failed    
# Calculate the percentage of failed files
failed_percentage = (failed_file_count / total_files) * 100 if total_files > 0 else 0

# Output the percentage of failed files
print(f"Percentage of failed files: {failed_percentage:.2f}%")