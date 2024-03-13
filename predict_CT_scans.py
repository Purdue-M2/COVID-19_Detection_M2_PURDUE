import torch
from torch.utils.data import DataLoader
from dataset import UniAttackDataset
from DFAD_model_base import DFADModel
import pandas as pd
from collections import Counter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = DFADModel()
model.to(device)
checkpoint_name = 'checkpoint_epoch.pt'
model.load_state_dict(torch.load(checkpoint_name, map_location=device))

# Prepare the validation dataset and loader
val_dataset = UniAttackDataset(
    hdf5_filename='task1_test.h5',
    dataset_name='test_features'
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the test function
def test(model, val_loader):
    model.eval()
    all_predicted_labels = []

    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)
            output = model(inputs).squeeze()
            probabilities = torch.sigmoid(output)
            predicted_labels = (probabilities.cpu().numpy() >= 0.5).astype(int)
            all_predicted_labels.extend(predicted_labels)
    
    return all_predicted_labels

# Execute the test function
predicted_labels = test(model, val_loader)

# File paths for the text files
image_paths_file = 'task1_test.txt'
# Save all predicted labels to a text file
prediction_file_path = 'task1_test_predictions.txt'

with open(prediction_file_path, 'w') as file:
    for label in predicted_labels:
        file.write(f"{label}\n")

print(f"Test predictions have been saved to {prediction_file_path}")

with open(image_paths_file, 'r') as f:
    image_paths = f.read().strip().split('\n')

with open(prediction_file_path, 'r') as f:
    labels = f.read().strip().split('\n')

# Now perform the majority vote
# Associate each image with its folder and label
folder_labels = {}
for path, label in zip(image_paths, predicted_labels):
    folder = path.split('/')[2]  # Adjust the index if necessary based on your path structure
    if folder not in folder_labels:
        folder_labels[folder] = []
    folder_labels[folder].append(str(label))

# Perform a majority vote for each folder and collect the results
results = []
for folder, lbls in folder_labels.items():
    most_common_label, _ = Counter(lbls).most_common(1)[0]
    results.append([folder, most_common_label])

# Save the results to a CSV file
results_df = pd.DataFrame(results, columns=['Folder', 'Label'])
results_df.to_csv('ct_series_task1_test_predictions.csv', index=False)

print("Results have been saved to ct_series_task1_test_predictions.csv")

