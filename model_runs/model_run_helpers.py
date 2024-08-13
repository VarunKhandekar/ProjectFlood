import torch
import torch.optim as optim
import csv
import datetime
from typing import Literal
from torch.utils.data import DataLoader
# import optuna
# from optuna.integration import PyTorchLightningPruningCallback
from dataloaders.convLSTM_dataset import *
from models.ConvLSTMSeparateBranches import *
from model_runs.model_evaluation_helpers import *
from visualisations.visualisation_helpers import *


def get_dataloader(label_file_name: Literal['training_labels_path', 'validation_labels_path', 'test_labels_path', 'training_validation_combo_path'], 
                   resolution: int, preceding_rainfall_days: int, forecast_rainfall_days: int, transform, 
                   batch_size: int, shuffle: bool, num_workers: int):
    dataset = FloodPredictionDataset(os.environ["PROJECT_FLOOD_DATA"], os.environ["PROJECT_FLOOD_CORE_PATHS"],
                                     label_file_name, resolution, preceding_rainfall_days, forecast_rainfall_days, transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def generate_model_name(base_name, date_today, **kwargs):
    # EXAMPLE: ConvLSTMSeparateBranches_num_epochs10_train_batch_size32_learning_rate0p0001_20240805
    # Replace periods in hyperparameter values with 'p'
    hyperparams_str = "_".join(f"{key}{str(value).replace('.', 'p')}" for key, value in kwargs.items())
    model_name = f"{base_name}_{hyperparams_str}_{date_today}"
    return model_name


def train_model(data_config_path: str, model,  criterion_type: str, optimizer_type: str, lr, num_epochs: int, device, 
                plot_training_images: bool, plot_losses: bool, 
                train_dataloader: DataLoader, val_dataloader: DataLoader = None):

    hyperparams = {
        'num_epochs': num_epochs,
        'train_batch_size': train_dataloader.batch_size,
        'learning_rate': lr,
        'preceding_rainfall_days': model.preceding_rainfall_days,
        'dropout_prob': model.dropout_prob,
        'output_channels': model.output_channels,
        'conv_block_layers': model.conv_block_layers,
        'convLSTM_layers': model.convLSTM_layers,
        'optimizer_type': optimizer_type,
        'criterion': criterion_type  
    }

    with open(data_config_path) as data_config_file:
        data_config = json.load(data_config_file)

    optimizer = getattr(optim, optimizer_type)(model.parameters(), lr=lr)
    criterion = getattr(nn, criterion_type)() #Default for BCELogitsLoss is mean reduction over the batch in question

    model = model.to(device)
    training_losses = []
    validation_losses = []
    epochs = []
    for epoch in range(1, num_epochs+1):
        # TRAINING
        model.train()
        training_epoch_loss = 0.0
        num_batches = 0
        for inputs, labels, flooded in train_dataloader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            outputs = model(inputs)            
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # last_outputs, last_labels = torch.sigmoid(outputs), labels # apply sigmoid for charting purposes
            last_outputs, last_labels, last_flooded = outputs, labels, flooded

            training_epoch_loss += loss.item()
            num_batches += 1

        # Save values down for loss chart plotting
        training_epoch_average_loss = training_epoch_loss/num_batches
        training_losses.append(training_epoch_average_loss)
        epochs.append(epoch)


        # COLLECT VALIDATION LOSSES
        if val_dataloader: # Check if we even want validation losses
            validation_epoch_average_loss = validate_model(model, val_dataloader, criterion, device)
            validation_losses.append(validation_epoch_average_loss)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {validation_epoch_average_loss:.4f}')

        # Save model snapshot
        if epoch % 1000 == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{model.name}_{epoch}.pt"), hyperparams)
    
    # Save final model            
    save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{model.name}_{epoch}.pt"), hyperparams)

    # PLOT EXAMPLE IMAGES ON TRAINING
    # Select only 4 samples from the last batch
    if plot_training_images:
        selected_outputs = last_outputs[:4]
        selected_labels = last_labels[:4]
        selected_labels_flooded = last_flooded[:4]
        image_examples_filename = os.path.join(data_config["training_plots_path"], f"outputs_vs_labels_{model.name}.png")
        plot_model_output_vs_label(selected_outputs, selected_labels, selected_labels_flooded, image_examples_filename)
        print("Training chart image saved!")
    
    # PLOT LOSS CHART
    # print(validation_losses)
    if plot_losses:
        losses = []
        losses.append(training_losses)
        if validation_losses:
            losses.append(validation_losses)
        loss_filename = os.path.join(data_config["loss_plots_path"], f"losschart_{model.name}.png")
        plot_loss_chart(losses, epochs, loss_filename, hyperparams)
        print("Loss chart image saved!")

    return model, epoch


def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets, flooded in dataloader:
            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, filepath, hyperparams):
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save({
        'model_type': model_to_save.__class__.__name__,
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparams': hyperparams
    }, filepath)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    hyperparams = checkpoint['hyperparams']
    if checkpoint['model_type'] == "ConvLSTMSeparateBranches":
        model = ConvLSTMSeparateBranches(hyperparams['preceding_rainfall_days'], 1,  
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


def evaluate_model(data_config_path, model, dataloader, criterion_str, device, epoch, model_run_date):
    model = model.to(device)
    model.eval()
    total_loss = 0
    total, correct = 0, 0
    metric_accumulator = {name: 0 for name in ["kl_div", "rmse", "mae", "psnr", "ssim", "fid"]}
    criterion = getattr(nn, criterion_str)()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            logits = torch.sigmoid(outputs)
            #TODO ADD CROPPING HERE
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

