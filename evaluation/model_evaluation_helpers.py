
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import csv
import json
import os
from models.ConvLSTMSeparateBranches import *
from models.ConvLSTMMerged import *
from evaluation.model_evaluation_helpers import *


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    hyperparams = checkpoint['hyperparams']
    print(checkpoint['hyperparams'].keys())
    if checkpoint['model_type'] == "ConvLSTMSeparateBranches":
        model = ConvLSTMSeparateBranches(hyperparams['precedingrainfall'], 1,  
                                         hyperparams['outputchannels'], 
                                         hyperparams['convblocklayers'],
                                         hyperparams['convLSTMlayers'],
                                         hyperparams['dropout'])
    elif checkpoint['model_type'] == "ConvLSTMMerged":
        model = ConvLSTMMerged(hyperparams['precedingrainfall'], 1,  
                               hyperparams['outputchannels'], 
                               hyperparams['convblocklayers'],
                               hyperparams['convLSTMlayers'],
                               hyperparams['dropout'])
        #TODO add the name as an attribute here?
    optimizer = getattr(optim, hyperparams['optim'])(model.parameters(), lr=hyperparams['lr'])

    #Set up model and optimizer with values from that checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], hyperparams


def KL_DivLoss(y_pred, y_true):
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    log_input = F.log_softmax(y_pred, dim=1)
    log_target = F.log_softmax(y_true, dim=1)
    return kl_loss(log_input, log_target)


def RMSELoss(y_pred, y_true):
    mse_loss = torch.nn.MSELoss(reduction="mean")
    return torch.sqrt(mse_loss(y_true, y_pred))


def MAELoss(y_pred, y_true):
    mae_loss = torch.nn.L1Loss(reduction="mean")
    return torch.sqrt(mae_loss(y_true, y_pred))  


def PSNR(y_pred, y_true):
    psnr = torchmetrics.PeakSignalNoiseRatio()
    return  psnr(y_pred, y_true)   


def SSIM(y_pred, y_true):
    if y_pred.dim() == 3:  # If shape is [B, H, W], add a channel dimension
        y_pred = y_pred.unsqueeze(1)  # Shape becomes [B, 1, H, W]
        y_true = y_true.unsqueeze(1)  # Shape becomes [B, 1, H, W]
    ssim = torchmetrics.StructuralSimilarityIndexMeasure()
    return ssim(y_pred, y_true)


def calculate_metrics(y_pred, y_true):
    metric_list = [
        ("kl_div", KL_DivLoss), 
        ("rmse", RMSELoss), 
        ("mae", MAELoss),
        ("psnr", PSNR),
        ("ssim", SSIM),
    ]
    metric_dict = {}
    for metric_name, metric_function in metric_list:
        metric_dict[metric_name] = metric_function(y_pred, y_true)
    return metric_dict



def evaluate_model(model, test_dataloader, criterion_str, device, mask_path, crop_right, crop_bottom, crop_left: int = 0, crop_top: int = 0):
    model = model.to(device)
    model.eval()
    criterion = getattr(torch.nn, criterion_str)()

    # Load the True/False mask
    mask = np.load(mask_path)

    thresholds = np.arange(0, 1.05, 0.05)
    confusion_matrices = {thr: np.zeros((2, 2)) for thr in thresholds}
    precision_scores = {thr: 0 for thr in thresholds}
    recall_scores = {thr: 0 for thr in thresholds}
    accuracy_scores = {thr: 0 for thr in thresholds}
    
    total_samples = 0

    total_loss = 0
    total_rmse = 0
    total_mae = 0
    total_psnr = 0
    total_ssim = 0

    with torch.no_grad():
        for inputs, labels, _ in test_dataloader: #batch size is first dim. (BXY)
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            outputs = model(inputs)

            # Crop the outputs and labels as required
            cropped_outputs = outputs[..., crop_top:crop_bottom, crop_left:crop_right]
            cropped_labels = labels[..., crop_top:crop_bottom, crop_left:crop_right]

            # Apply mask (retain False values)
            mask_torch = torch.from_numpy(mask).to(device)
            mask_torch = mask_torch.unsqueeze(0)
            mask_torch = mask_torch.expand(cropped_outputs.size(0), -1, -1)

            masked_outputs = cropped_outputs[~mask_torch]
            masked_labels = cropped_labels[~mask_torch]

            # 1. Calculate PSNR and SSIM on cropped (but not masked) images
            psnr_value = PSNR(cropped_outputs, cropped_labels)
            ssim_value = SSIM(cropped_outputs, cropped_labels)
            total_psnr += psnr_value.item()
            total_ssim += ssim_value.item()

            # 2. Calculate RMSE and MAE on masked images
            rmse_loss = RMSELoss(masked_outputs, masked_labels)
            mae_loss = MAELoss(masked_outputs, masked_labels)

            total_rmse += rmse_loss.item()
            total_mae += mae_loss.item()

            # Calculate the total loss
            loss = criterion(masked_outputs, masked_labels).item()
            total_loss += loss
            total_samples += masked_labels.size(0)

            # Apply different thresholds and calculate confusion matrices
            for thr in thresholds:
                binary_output = (masked_outputs > thr).float()
                binary_output = binary_output.cpu().numpy().flatten()
                binary_labels = masked_labels.cpu().numpy().flatten()

                cm = confusion_matrix(binary_labels, binary_output, labels=[0, 1]) #tn, fp, fn, tp
                confusion_matrices[thr] += cm
                precision_scores[thr] += precision_score(binary_labels, binary_output, zero_division=0)
                recall_scores[thr] += recall_score(binary_labels, binary_output, zero_division=0)

                # Calculate accuracy at this threshold
                correct_predictions = (binary_output == binary_labels).sum()
                accuracy_scores[thr] += correct_predictions

    # Calculate averages
    avg_rmse = total_rmse / len(test_dataloader)
    avg_mae = total_mae / len(test_dataloader)
    avg_psnr = total_psnr / len(test_dataloader)
    avg_ssim = total_ssim / len(test_dataloader)
    average_loss = total_loss / total_samples

    # Calculate the final precision and recall
    for thr in thresholds:
        precision_scores[thr] /= total_samples
        recall_scores[thr] /= total_samples
        accuracy_scores[thr] /= total_samples

    metric_accumulator = {
        'average_rmse': avg_rmse,
        'average_mae': avg_mae,
        'average_psnr': avg_psnr,
        'average_ssim': avg_ssim,
        'average_loss': average_loss,
        'confusion_matrices': confusion_matrices,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'accuracy_scores': accuracy_scores
    }

    return metric_accumulator














# def evaluate_model(data_config_path, model, dataloader, criterion_str, device):
#     model = model.to(device)
#     model.eval()
#     total_loss = 0
#     total, correct = 0, 0
#     metric_accumulator = {name: 0 for name in ["kl_div", "rmse", "mae", "psnr", "ssim", "fid"]}
#     criterion = getattr(nn, criterion_str)()

#     with torch.no_grad():
#         for inputs, labels, flooded in dataloader:
#             inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
#             outputs = model(inputs)
#             total_loss += criterion(outputs, labels).item()
#             #TODO ADD CROPPING HERE
#             predicted = outputs > 0.5
#             total += labels.size(0)
#             correct += (predicted == labels.to(device)).sum().item()

#             # Calculate metrics for each batch and accumulate
#             batch_metrics = calculate_metrics(outputs, labels)
#             for key in metric_accumulator:
#                 metric_accumulator[key] += batch_metrics[key]

#     # Average the accumulated metrics over all batches
#     for key in metric_accumulator:
#         metric_accumulator[key] /= len(dataloader)

#     accuracy = 100 * correct / total if total > 0 else 0
#     average_loss = total_loss / len(dataloader)

#     with open(data_config_path) as data_config_file:
#         data_config = json.load(data_config_file)

#     with open(os.path.join(data_config["model_results_path"], f"{model.name}_evaluation_results.csv"), mode='w', newline='') as file:
#         writer = csv.writer(file)
#         # Write headers
#         headers = ['Metric', 'Value']
#         writer.writerow(headers)
#         # Write data
#         writer.writerow(['Accuracy', accuracy])
#         writer.writerow(['Average Loss', average_loss])
#         for key, value in metric_accumulator.items():
#             writer.writerow([key, value])
#     return accuracy, average_loss, metric_accumulator