
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from models.ConvLSTMSeparateBranches import *
from models.ConvLSTMMerged import *
from evaluation.model_evaluation_helpers import *


def load_checkpoint(filepath: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(filepath, map_location=device)
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


def KL_DivLoss(y_pred: torch.Tensor, y_true: torch.Tensor):
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    log_input = F.log_softmax(y_pred, dim=1)
    log_target = F.log_softmax(y_true, dim=1)
    return kl_loss(log_input, log_target)


def RMSELoss(y_pred: torch.Tensor, y_true: torch.Tensor):
    mse_loss = torch.nn.MSELoss(reduction="mean")
    return torch.sqrt(mse_loss(y_true, y_pred))


def MAELoss(y_pred: torch.Tensor, y_true: torch.Tensor):
    mae_loss = torch.nn.L1Loss(reduction="mean")
    return torch.sqrt(mae_loss(y_true, y_pred))  


def PSNR(y_pred: torch.Tensor, y_true: torch.Tensor):
    psnr = torchmetrics.PeakSignalNoiseRatio()
    return  psnr(y_pred, y_true)   


def SSIM(y_pred: torch.Tensor, y_true: torch.Tensor):
    if y_pred.dim() == 3:  # If shape is [B, H, W], add a channel dimension
        y_pred = y_pred.unsqueeze(1)  # Shape becomes [B, 1, H, W]
        y_true = y_true.unsqueeze(1)  # Shape becomes [B, 1, H, W]
    ssim = torchmetrics.StructuralSimilarityIndexMeasure()
    return ssim(y_pred, y_true)


def SSIM_structural(y_pred: torch.Tensor, y_true: torch.Tensor, C3: float = 1e-3, epsilon:float = 1e-10):
    if y_pred.dim() == 3:  # If shape is [B, H, W], add a channel dimension
        y_pred = y_pred.unsqueeze(1)  # Shape becomes [B, 1, H, W]
        y_true = y_true.unsqueeze(1)  # Shape becomes [B, 1, H, W]
    # Mean of the images
    mu1 = F.avg_pool2d(y_pred, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(y_true, kernel_size=11, stride=1, padding=5)
    
    # Standard deviation
    sigma1_sq = F.avg_pool2d(y_pred ** 2, kernel_size=11, stride=1, padding=5) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(y_true ** 2, kernel_size=11, stride=1, padding=5) - mu2 ** 2

    # Add epsilon to avoid negative values or division by zero
    sigma1 = torch.sqrt(sigma1_sq.clamp(min=epsilon))
    sigma2 = torch.sqrt(sigma2_sq.clamp(min=epsilon))
    
    # Covariance
    sigma12 = F.avg_pool2d(y_pred * y_true, kernel_size=11, stride=1, padding=5) - mu1 * mu2
    
    # Structural component
    s = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    return s.mean()


def evaluate_model(model, test_dataloader, criterion_str: str, device, mask_path: str, crop_right: int, crop_bottom: int, crop_left: int = 0, crop_top: int = 0):
    model = model.to(device)
    model.eval()
    criterion = getattr(torch.nn, criterion_str)()

    mask = np.load(mask_path) # Load the True/False mask

    thresholds = np.arange(0, 1.05, 0.05)
    confusion_matrices = {thr: np.zeros((2, 2)) for thr in thresholds}
    
    accuracy_scores = {thr: 0 for thr in thresholds}
    
    total_samples = 0
    total_masked_samples = 0

    total_loss = 0
    total_rmse = 0
    total_mae = 0
    total_psnr = 0
    total_ssim = 0
    total_ssim_struct = 0

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

            # 1. Calculate PSNR, SSIM, RMSE and MAE on cropped (but not masked) images
            psnr_value = PSNR(cropped_outputs, cropped_labels)
            ssim_value = SSIM(cropped_outputs, cropped_labels)
            ssim_struct_value = SSIM_structural(cropped_outputs, cropped_labels)
            total_psnr += psnr_value.item()
            total_ssim += ssim_value.item()
            total_ssim_struct += ssim_struct_value.item()

            rmse_loss = RMSELoss(cropped_outputs, cropped_labels)
            mae_loss = MAELoss(cropped_outputs, cropped_labels)

            total_rmse += rmse_loss.item()
            total_mae += mae_loss.item()

            # Calculate the total loss
            loss = criterion(cropped_outputs, cropped_labels).item()
            total_loss += loss
            total_samples += cropped_labels.size(0)
            total_masked_samples += masked_labels.size(0)

            # 2. Apply different thresholds and calculate confusion matrices
            for thr in thresholds:
                binary_output = (masked_outputs > thr).float()
                binary_output = binary_output.cpu().numpy().flatten()
                binary_labels = masked_labels.cpu().numpy().flatten()

                cm = confusion_matrix(binary_labels, binary_output, labels=[0, 1]) #tn, fp, fn, tp, does counts
                confusion_matrices[thr] += cm # element-wise addition

                # Calculate accuracy at this threshold
                correct_predictions = (binary_output == binary_labels).sum()
                accuracy_scores[thr] += correct_predictions

    # Calculate averages
    avg_rmse = total_rmse / len(test_dataloader)
    avg_mae = total_mae / len(test_dataloader)
    avg_psnr = total_psnr / len(test_dataloader)
    avg_ssim = total_ssim / len(test_dataloader)
    avg_ssim_struct = total_ssim_struct / len(test_dataloader)
    average_loss = total_loss / len(test_dataloader)


    # Calculate average accuracy for each threshold
    for thr in thresholds:
        accuracy_scores[thr] /= total_masked_samples

    metric_accumulator = {
        'average_rmse': avg_rmse,
        'average_mae': avg_mae,
        'average_psnr': avg_psnr,
        'average_ssim': avg_ssim,
        'average_ssim_struct': avg_ssim_struct,
        'average_loss': average_loss,
        'confusion_matrices': confusion_matrices,
        'accuracy_scores': accuracy_scores
    }
    
    # Calculate the final precision, recall, F1 and false positive rate
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}
    false_positive_rates = {}
    for thr, cm in metric_accumulator['confusion_matrices'].items():
        tn, fp, fn, tp = cm.ravel()  # Unpack the 2x2 confusion matrix
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        
        precision_scores[thr] = precision
        recall_scores[thr] = recall
        f1_scores[thr] = (2*precision*recall)/(precision+recall) if (precision + recall) > 0 else np.nan

        false_positive_rates[thr] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    
    metric_accumulator['precision_scores'] = precision_scores
    metric_accumulator['recall_scores'] = recall_scores
    metric_accumulator['f1_scores'] = f1_scores
    metric_accumulator['false_positive_rates'] = false_positive_rates

    # print('Accuracy scores:',  metric_accumulator['accuracy_scores'])
    # print('Precision scores:',  metric_accumulator['precision_scores'])
    # print('Recall scores:',  metric_accumulator['recall_scores'])
    # print('F1 scores:',  metric_accumulator['f1_scores'])
    # print('False Positive scores:',  metric_accumulator['false_positive_rates'])

    return metric_accumulator


def save_metrics_to_csv(metric_accumulators: list, filename: str, model_names: list):
    data = {}
    for metric_accumulator in metric_accumulators:
        for key, value in metric_accumulator.items():
            if key not in data:
                data[key] = []
            data[key].append(value)

    df = pd.DataFrame(data, index=model_names).T
    df.to_csv(filename)


def collect_images(model, dataloader, size):
    flooded_images, non_flooded_images = [], []
    for inputs, targets, flooded in dataloader:
        outputs = model(inputs)
        for i in range(len(flooded)):
            if flooded[i] == 1 and len(flooded_images) < size:
                flooded_images.append((outputs[i], targets[i], flooded[i]))
            elif flooded[i] == 0 and len(non_flooded_images) < size:
                non_flooded_images.append((outputs[i], targets[i], flooded[i]))
            if len(flooded_images) >= size and len(non_flooded_images) >= size:
                return flooded_images, non_flooded_images
    return flooded_images, non_flooded_images # list of tuples