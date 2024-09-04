
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


def load_checkpoint(filepath: str) -> tuple:
    """
    Load a pytorch model checkpoint from a file, including the model, optimizer state, epoch, and hyperparameters.

    Args:
        filepath (str): Path to the checkpoint file to load.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The model initialized and loaded with the checkpoint state.
            - optimizer (torch.optim.Optimizer): The optimizer loaded with the checkpoint state.
            - epoch (int): The epoch number at which the checkpoint was saved.
            - hyperparams (dict): A dictionary of hyperparameters used to configure the model and optimizer.

    """
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


def KL_DivLoss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the Kullback-Leibler (KL) divergence loss between two tensors.

    Args:
        y_pred (torch.Tensor): Predicted tensor of log probabilities.
        y_true (torch.Tensor): Ground truth tensor of log probabilities.

    Returns:
        torch.Tensor: The batch mean KL divergence loss between `y_pred` and `y_true`.

    """
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    log_input = F.log_softmax(y_pred, dim=1)
    log_target = F.log_softmax(y_true, dim=1)
    return kl_loss(log_input, log_target)


def RMSELoss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the Root Mean Squared Error (RMSE) loss between predicted and true tensors.

    Args:
        y_pred (torch.Tensor): Predicted tensor.
        y_true (torch.Tensor): Ground truth tensor.

    Returns:
        torch.Tensor: The RMSE loss between `y_pred` and `y_true`.

    """
    mse_loss = torch.nn.MSELoss(reduction="mean")
    return torch.sqrt(mse_loss(y_true, y_pred))


def MAELoss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the Mean Absolute Error (MAE) loss between predicted and true tensors.

    Args:
        y_pred (torch.Tensor): Predicted tensor.
        y_true (torch.Tensor): Ground truth tensor.

    Returns:
        torch.Tensor: The square root of the MAE loss between `y_pred` and `y_true`.

    """
    mae_loss = torch.nn.L1Loss(reduction="mean")
    return torch.sqrt(mae_loss(y_true, y_pred))  


def PSNR(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between predicted and true tensors.

    Args:
        y_pred (torch.Tensor): Predicted tensor.
        y_true (torch.Tensor): Ground truth tensor.

    Returns:
        torch.Tensor: The PSNR value between `y_pred` and `y_true`.

    """
    psnr = torchmetrics.PeakSignalNoiseRatio()
    return  psnr(y_pred, y_true)   


def SSIM(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the Structural Similarity Index (SSIM) between predicted and true images.

    Args:
        y_pred (torch.Tensor): The predicted tensor, with shape [B, H, W] or [B, C, H, W].
        y_true (torch.Tensor): The ground truth tensor, with shape [B, H, W] or [B, C, H, W].

    Returns:
        torch.Tensor: The SSIM value between `y_pred` and `y_true`.

    """
    if y_pred.dim() == 3:  # If shape is [B, H, W], add a channel dimension
        y_pred = y_pred.unsqueeze(1)  # Shape becomes [B, 1, H, W]
        y_true = y_true.unsqueeze(1)  # Shape becomes [B, 1, H, W]
    ssim = torchmetrics.StructuralSimilarityIndexMeasure()
    return ssim(y_pred, y_true)


def SSIM_structural(y_pred: torch.Tensor, y_true: torch.Tensor, C3: float = 1e-3, epsilon:float = 1e-10) -> torch.Tensor:
    """
    Compute the structural component of the Structural Similarity Index (SSIM) between predicted and true images.

    Args:
        y_pred (torch.Tensor): The predicted tensor, with shape [B, H, W] or [B, C, H, W].
        y_true (torch.Tensor): The ground truth tensor, with shape [B, H, W] or [B, C, H, W].
        C3 (float, optional): Stability constant for the structural component. Default is 1e-3.
        epsilon (float, optional): A small value to avoid division by zero. Default is 1e-10.

    Returns:
        torch.Tensor: The mean structural component of SSIM between `y_pred` and `y_true`.

    """
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


def evaluate_model(model: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader, 
                   criterion_str: str, device: torch.device, mask_path: str, 
                   crop_right: int, crop_bottom: int, crop_left: int = 0, crop_top: int = 0) -> dict:
    """
    Evaluate a model's performance on a test dataset by calculating various metrics such as RMSE, MAE, PSNR, SSIM, and confusion matrices across different thresholds.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_dataloader (torch.utils.data.DataLoader): The DataLoader providing the test dataset.
        criterion_str (str): The name of the loss function to be used (e.g., 'MSELoss').
        device (torch.device): The device to run the evaluation on (CPU or CUDA).
        mask_path (str): Path to the numpy file containing the mask (True/False values).
        crop_right (int): The right boundary for cropping the outputs and labels.
        crop_bottom (int): The bottom boundary for cropping the outputs and labels.
        crop_left (int, optional): The left boundary for cropping the outputs and labels. Default is 0.
        crop_top (int, optional): The top boundary for cropping the outputs and labels. Default is 0.

    Returns:
        dict: A dictionary containing various performance metrics, including:
            - 'average_rmse': Average Root Mean Squared Error (RMSE) across batches.
            - 'average_mae': Average Mean Absolute Error (MAE) across batches.
            - 'average_psnr': Average Peak Signal-to-Noise Ratio (PSNR) across batches.
            - 'average_ssim': Average Structural Similarity Index Measure (SSIM) across batches.
            - 'average_ssim_struct': Average structural component of SSIM across batches.
            - 'average_loss': Average loss across batches.
            - 'confusion_matrices': Confusion matrices for different thresholds.
            - 'accuracy_scores': Accuracy scores for different thresholds.
            - 'precision_scores': Precision scores for different thresholds.
            - 'recall_scores': Recall scores for different thresholds.
            - 'f1_scores': F1 scores for different thresholds.
            - 'false_positive_rates': False positive rates for different thresholds.

    """
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


def save_metrics_to_csv(metric_accumulators: list, filename: str, model_names: list) -> None:
    """
    Save accumulated evaluation metrics for multiple models to a CSV file.

    Args:
        metric_accumulators (list): A list of dictionaries, where each dictionary contains the metrics accumulated for a model.
        filename (str): The path and name of the CSV file to save the metrics.
        model_names (list): A list of model names corresponding to the metric accumulators, used as the index in the CSV.

    Returns:
        None: The function saves the metrics to a CSV file.

    """
    data = {}
    for metric_accumulator in metric_accumulators:
        for key, value in metric_accumulator.items():
            if key not in data:
                data[key] = []
            data[key].append(value)

    df = pd.DataFrame(data, index=model_names).T
    df.to_csv(filename)


def collect_images(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, size: int) -> tuple:
    """
    Collect a specified number of flooded and non-flooded images (output, target, and label) from the dataloader using a model.

    Args:
        model (torch.nn.Module): The model used to generate outputs from the inputs.
        dataloader (torch.utils.data.DataLoader): The dataloader providing batches of inputs, targets, and labels indicating flooded or non-flooded areas.
        size (int): The number of flooded and non-flooded images to collect.

    Returns:
        tuple: A tuple containing two lists:
            - `flooded_images`: A list of tuples (output, target, flooded label) for flooded images.
            - `non_flooded_images`: A list of tuples (output, target, non-flooded label) for non-flooded images.

    """
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
    return flooded_images, non_flooded_images