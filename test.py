import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# import torch.optim as optim

from torch.utils.data import DataLoader

from dataset import UniAttackDataset
from DFAD_model_base import DFADModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = DFADModel()
model.to(device)
checkpoint_name = 'checkpoints_teach/'
model.load_state_dict(torch.load(checkpoint_name))


val_dataset = UniAttackDataset(
    hdf5_filename='task2_non_clip_vit.h5',
    dataset_name='train_features'
)

val_loader = DataLoader(val_dataset, batch_size=32,shuffle=False)


model.eval()

def test(model, test_loader):
    model.eval()
    all_predicted_labels = []

    with torch.no_grad():
        for inputs in tqdm(test_loader):  # Assuming the test_loader does not provide labels
            inputs = inputs.to(device)
            output = model(inputs).squeeze()
            probabilities = torch.sigmoid(output)
            predicted_labels = (probabilities.cpu().numpy() >= 0.5 ).astype(int)
            all_predicted_labels.extend(predicted_labels)
    
    return all_predicted_labels


predicted_labels = test(model, val_loader)
# Save all predicted labels to a text file
prediction_file_path = 'test_predictions.txt'
with open(prediction_file_path, 'w') as file:
    for label in predicted_labels:
        file.write(f"{label}\n")

print(f"Test predictions have been saved to {prediction_file_path}")