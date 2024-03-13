import torch
import torch.nn as nn
from bypass_bn import enable_running_stats, disable_running_stats
from sam import SAM
from torch.utils.data import DataLoader
from tqdm import tqdm
# import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix,accuracy_score, roc_auc_score, f1_score

from scipy import optimize
from dataset import UniAttackDataset
from DFAD_model_base import DFADModel

import os


checkpoint_dir = 'checkpoints_task1'
metrics_file_path = 'traing_log_task1.txt' 
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def threshplus_tensor(x):
    y = x.clone()
    pros = torch.nn.ReLU()
    z = pros(y)
    return z

def search_func(losses, alpha):
    return lambda x: x + (1.0/alpha)*(threshplus_tensor(losses-x).mean().item())

def searched_lamda_loss(losses, searched_lamda, alpha):
    return searched_lamda + ((1.0/alpha)*torch.mean(threshplus_tensor(losses-searched_lamda))) 

def calculate_L_AUC(P_scores, N_scores, gamma, p):
    # Convert scores to column and row vectors respectively
    P_scores = P_scores.unsqueeze(1)  # Make it a column vector
    N_scores = N_scores.unsqueeze(0)  # Make it a row vector

    # Compute the margin matrix in a vectorized form
    margin_matrix = P_scores - N_scores - gamma

    # Apply the ReLU-like condition and raise to power p
    loss_matrix = torch.where(margin_matrix < 0, (-margin_matrix) ** p, torch.zeros_like(margin_matrix))

    # Compute the final L_AUC by averaging over all elements
    L_AUC = loss_matrix.mean()

    return L_AUC



def train_epoch(model, optimizer, scheduler, criterion, train_loader,loss_type):
    model.train()
    total_loss_accumulator = 0
    all_labels = []
    all_predictions = []
    alpha_cvar = 0.5


    def calculate_loss(output, labels, loss_type, criterion):
    
        loss_ce = criterion(output, labels)
        # Directly return loss_ce for 'erm' loss type
        if loss_type == 'erm':
            return loss_ce

        # For 'dag' loss types, perform additional computations
        if loss_type in ['dag']:
            chi_loss_np = search_func(loss_ce, alpha_cvar)
            cutpt = optimize.fminbound(chi_loss_np, np.min(loss_ce.cpu().detach().numpy()) - 1000.0, np.max(loss_ce.cpu().detach().numpy()))
            loss = searched_lamda_loss(loss_ce, cutpt, alpha_cvar)

        return loss

    for inputs, labels in tqdm(train_loader, desc="Progress"):

        inputs, labels = inputs.to(device), labels.to(device)


        enable_running_stats(model)
        output = model(inputs).squeeze()
        total_loss = calculate_loss(output, labels,loss_type,criterion)  
        total_loss.backward()
        optimizer.first_step(zero_grad=True)

        disable_running_stats(model) 
        output = model(inputs).squeeze()
        total_loss = calculate_loss(output, labels,loss_type,criterion)
        total_loss.backward()
        optimizer.second_step(zero_grad=True)

        total_loss_accumulator += total_loss.item()   

        predictions = torch.sigmoid(output) >= 0.5
        # Accumulate labels and predictions for metrics calculation
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    scheduler.step()

    # Convert accumulated labels and predictions to NumPy arrays for metric computation
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Compute metrics
    acc = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    # For TPR (Sensitivity) and FOR calculation, you might need confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    tpr = tp / (tp + fn)  # True Positive Rate
    fpr = fp / (fp + tn)  # False Positive Rate
    return  total_loss_accumulator / len(train_loader), acc, tpr, fpr, f1

def evaluate(model,val_loader):
    model.eval()
    all_labels = []
    all_probabilities = []  # Use this to store probabilities for all samples
  


    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs).squeeze()
            probabilities = torch.sigmoid(output)
            
            # Collect probabilities for each batch
            all_probabilities.append(probabilities.cpu().numpy())  # Store as NumPy array
            all_labels.extend(labels.cpu().numpy())


    # Concatenate probabilities from all batches
    all_probabilities = np.concatenate(all_probabilities)
    predicted_labels = all_probabilities >= 0.5  # Convert probabilities to binary predictions

    # Calculate metrics
    acc = accuracy_score(all_labels, predicted_labels)
    auc = roc_auc_score(all_labels, all_probabilities)

    # Calculate F1 scores
    f1_positive = f1_score(all_labels, predicted_labels, pos_label=1)
    f1_negative = f1_score(all_labels, predicted_labels, pos_label=0)
    f1_macro = f1_score(all_labels, predicted_labels, average='macro')
    

    return  acc, auc, f1_positive, f1_negative, f1_macro



def model_trainer(loss_type, batch_size=64, num_epochs=64):
    seed = 5
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Move model to GPU
    model = DFADModel().to(device)
    train_dataset = UniAttackDataset(
    hdf5_filename='train_clip.h5',
    labels_filename='train.txt',
    dataset_name='train_features'
)
    val_dataset = UniAttackDataset(
    hdf5_filename='val_clip.h5',
    labels_filename='val.txt',
    dataset_name='val_features'
)
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=32, shuffle=False)


    # Prepare data loaders
    if loss_type == 'erm':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    
    # Initialize optimizer and scheduler
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    # Initialize the learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=num_epochs / 4, eta_min=1e-5)  # eta_min is the minimum lr


    # trian and evaluate
    with open(metrics_file_path, 'w') as metrics_file:
        for epoch in range(num_epochs):
            epoch_str = str(epoch).zfill(4)
            print(epoch_str)
            train_loss, train_acc,  train_tpr, train_fpr, train_f1 = train_epoch(model, optimizer, scheduler, criterion,train_loader,loss_type)

            # accuracy, auc= evaluate(model, criterion, val_loader)
            acc, auc, f1_positive, f1_negative, f1_macro = evaluate(model, val_loader)
    

            # Write metrics to console and file
            metrics_str = (
                f'Epoch: {epoch_str}\n'
                f'Train Loss: {train_loss:.6f}, ACC: {train_acc:.6f},  TPR: {train_tpr:.6f}, FPR: {train_fpr:.6f}, F1: {train_f1:.6f}\n'
                f"Val ACC: {acc}, AUC: {auc}, f1_positive: {f1_positive}, f1_negative: {f1_negative}, F1: {f1_macro}\n\n"
            )
            print(metrics_str)
            metrics_file.write(metrics_str)

            print()


            # save checkpoints
            checkpoint_name = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_name)




if __name__ == '__main__':

    model_trainer(loss_type='dag', batch_size=32, num_epochs=32)
