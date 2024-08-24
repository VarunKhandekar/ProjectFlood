
import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as T
import csv
import json
import os
from models.ConvLSTMSeparateBranches import *
from models.ConvLSTMMerged import *
from final_evaluation.model_evaluation_helpers import *


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    hyperparams = checkpoint['hyperparams']
    if checkpoint['model_type'] == "ConvLSTMSeparateBranches":
        model = ConvLSTMSeparateBranches(hyperparams['preceding_rainfall_days'], 1,  
                                         hyperparams['output_channels'], 
                                         hyperparams['conv_block_layers'],
                                         hyperparams['convLSTM_layers'],
                                         hyperparams['dropout_prob'])
    elif checkpoint['model_type'] == "ConvLSTMMerged":
        model = ConvLSTMMerged(hyperparams['preceding_rainfall_days'], 1,  
                                         hyperparams['output_channels'], 
                                         hyperparams['conv_block_layers'],
                                         hyperparams['convLSTM_layers'],
                                         hyperparams['dropout_prob'])
        #TODO add the name as an attribute here?
    optimizer = getattr(optim, hyperparams['optimizer_type'])(model.parameters(), lr=hyperparams['learning_rate'])

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
    ssim = torchmetrics.StructuralSimilarityIndexMeasure()
    return ssim(y_pred, y_true)


def FID(y_pred, y_true):
    from torchmetrics.image.fid import FrechetInceptionDistance
    
    fid = FrechetInceptionDistance(feature=64, normalize=True)
    fid.update(y_true, real=True)
    fid.update(y_pred, real=False)
    return fid.compute()


def calculate_metrics(y_pred, y_true):
    metric_list = [
        ("kl_div", KL_DivLoss), 
        ("rmse", RMSELoss), 
        ("mae", MAELoss),
        ("psnr", PSNR),
        ("ssim", SSIM),
        ("fid", FID)
    ]
    metric_dict = {}
    for metric_name, metric_function in metric_list:
        metric_dict[metric_name] = metric_function(y_pred, y_true)
    return metric_dict


def evaluate_model(data_config_path, model, dataloader, criterion_str, device, epoch, model_run_date):
    model = model.to(device)
    model.eval()
    total_loss = 0
    total, correct = 0, 0
    metric_accumulator = {name: 0 for name in ["kl_div", "rmse", "mae", "psnr", "ssim", "fid"]}
    criterion = getattr(nn, criterion_str)()

    with torch.no_grad():
        for inputs, labels, flooded in dataloader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            #TODO ADD CROPPING HERE
            predicted = outputs > 0.5
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

            # Calculate metrics for each batch and accumulate
            batch_metrics = calculate_metrics(outputs, labels)
            for key in metric_accumulator:
                metric_accumulator[key] += batch_metrics[key]

    # Average the accumulated metrics over all batches
    for key in metric_accumulator:
        metric_accumulator[key] /= len(dataloader)

    accuracy = 100 * correct / total if total > 0 else 0
    average_loss = total_loss / len(dataloader)

    with open(data_config_path) as data_config_file:
        data_config = json.load(data_config_file)

    with open(os.path.join(data_config["model_results_path"], f"{model.name}_{epoch}_{model_run_date}_evaluation_results.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        headers = ['Metric', 'Value']
        writer.writerow(headers)
        # Write data
        writer.writerow(['Accuracy', accuracy])
        writer.writerow(['Average Loss', average_loss])
        for key, value in metric_accumulator.items():
            writer.writerow([key, value])
    return accuracy, average_loss, metric_accumulator