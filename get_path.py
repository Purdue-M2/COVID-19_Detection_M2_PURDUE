import os


txt1_path = 'task2_non.txt'  # This file contains image paths
txt2_path = 'test_predictions.txt'  # This file contains labels
output_path = 'task2_non_pred.txt'  # Path for the new combined file

# Open both files for reading and the output file for writing
with open(txt1_path, 'r') as txt1, open(txt2_path, 'r') as txt2, open(output_path, 'w') as output:
    for path_line, label_line in zip(txt1, txt2):
        # Strip newline characters and combine path and label
        combined_line = f"{path_line.strip()}\t{label_line.strip()}\n"
        # Write the combined line to the output file
        output.write(combined_line)

print(f"Combined file has been created at {output_path}")
