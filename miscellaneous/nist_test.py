import numpy
from nistrng import *
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'output.bin')

def read_binary_file(file_path):
    with open(file_path, 'rb') as file:
        data = file.read()
    return data

if __name__ == "__main__":
    binary_data = read_binary_file(file_path)

    bits = numpy.fromiter(binary_data, dtype='uint8')
    
    binary_sequence: numpy.ndarray = pack_sequence(bits)
    # Check the eligibility of the test and generate an eligible battery from the default NIST-sp800-22r1a battery
    eligible_battery = check_eligibility_all_battery(binary_sequence, SP800_22R1A_BATTERY)

    # Print the eligible tests
    print("Eligible test from NIST-SP800-22r1a:")
    for name in eligible_battery.keys():
        print("-" + name)

    # Test the sequence on the eligible tests
    results = run_all_battery(binary_sequence, eligible_battery, False)

    # Print results one by one
    print("Test results:")
    for result, elapsed_time in results:
        if result.passed:
            print("- PASSED - score: " + str(numpy.round(result.score, 3)) +
                  " - " + result.name + " - elapsed time: " + str(elapsed_time) + " ms")
        else:
            print("- FAILED - score: " + str(numpy.round(result.score, 3)) +
                  " - " + result.name + " - elapsed time: " + str(elapsed_time) + " ms")
