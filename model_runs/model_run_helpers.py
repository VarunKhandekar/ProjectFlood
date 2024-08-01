import torch
import torch.optim as optim
import csv
from typing import Literal
from datetime import date
from torch.utils.data import DataLoader
from dataloaders.convLSTM_dataloader import *
from models.convLSTM_separate_branches import *
from model_runs.model_evaluation_helpers import *


def get_dataloader(label_file_name: Literal['training_labels_path', 'validation_labels_path', 'test_labels_path'], 
                   resolution: int, preceding_rainfall_days: int, forecast_rainfall_days: int, transform, 
                   batch_size: int, shuffle: bool, num_workers: int):
    dataset = FloodPredictionDataset(os.environ["PROJECT_FLOOD_DATA"], label_file_name, resolution, preceding_rainfall_days, forecast_rainfall_days, transform)
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers)
    return dataloader


# def build_model(model_type: Literal['convLSTM_separate_branches', 'vae', 'ddpm'], preceding_rainfall_days: int, forecast_rainfall_days: int, dropout_prob: float = 0.0):
#     if model_type == 'convLSTM_separate_branches':
#         model = ConvLSTMCombinedModel(preceding_rainfall_days, forecast_rainfall_days, dropout_prob)
    
#     elif model_type == 'vae':
#         pass
    
#     elif model_type == 'ddpm':
#         pass
    
#     return model


def train_model(data_config_path: str, model, dataloader, criterion, optimizer_type, lr, num_epochs, device, model_run_date):
    
    with open(data_config_path) as data_config_file:
        data_config = json.load(data_config_file)

    optimizer = optimizer_type(model.parameters(), lr=lr)
    model = model.to_device()
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        if epoch % 100 == 0:
            hyperparams = {
                            'learning_rate': lr,
                            'train_batch_size': dataloader.batch_size,
                            'optimizer_type': optimizer.__class__.__name__,  # Extracting the class name as a string
                            'num_epochs': num_epochs,
                            'preceding_rainfall_days': model.preceding_rainfall_days,
                            'dropout_prob': model.dropout_prob,
                            'criterion': criterion.__class__.__name__  # Extracting the class name as a string
                        }
            save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{model.name}_{epoch}_{model_run_date}.pt"), hyperparams)
    save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{model.name}_{epoch}_{model_run_date}.pt"), hyperparams)
    return model, epoch


def save_checkpoint(model, optimizer, epoch, filepath, hyperparams):
    torch.save({
        'model_type': model.name,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparams': hyperparams
    }, filepath)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    hyperparams = checkpoint['hyperparams']
    if checkpoint['model_type'] == "convLSTM_separate_branches":
        model = ConvLSTMSeparateBranches(hyperparams['preceding_rainfall_days'], 1,  hyperparams['dropout_prob'])
    optimizer = getattr(optim, hyperparams['optimizer_type'])(model.parameters(), lr=hyperparams['learning_rate'])

    #Set up model and optimizer with values from that checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], hyperparams



def evaluate_model(data_config_path, model, dataloader, criterion, device, epoch, model_run_date):
    model.eval()
    total_loss = 0
    total, correct = 0, 0
    metric_accumulator = {name: 0 for name in ["kl_div", "rmse", "mae", "psnr", "ssim", "fid"]}

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            logits = torch.sigmoid(outputs)
            predicted = logits > 0.5
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

    with open(os.path.join(data_config["model_results_path"], f"{model.name}_{epoch}_{model_run_date}_evaluarion_results.csv"), mode='w', newline='') as file:
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


