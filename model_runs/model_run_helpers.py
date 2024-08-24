import torch
import torch.optim as optim

import datetime
from typing import Literal
from torch.utils.data import DataLoader
# import optuna
# from optuna.integration import PyTorchLightningPruningCallback
from dataloaders.convLSTM_dataset import *
from models.ConvLSTMSeparateBranches import *
from models.ConvLSTMMerged import *
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
                train_dataloader: DataLoader, val_dataloader: DataLoader = None, is_final: bool = False):

    hyperparams = {
        'epochs': num_epochs,
        'batchsize': train_dataloader.batch_size,
        'lr': lr,
        'precedingrainfall': model.preceding_rainfall_days,
        'dropout': model.dropout_prob,
        'outputchannels': model.output_channels,
        'convblocklayers': model.conv_block_layers,
        'convLSTMlayers': model.convLSTM_layers,
        'optim': optimizer_type,
        'criterion': criterion_type,
        'transforms': True if train_dataloader.dataset.transform else False,
        'res': train_dataloader.dataset.resolution
    }
    # print(hyperparams)

    with open(data_config_path) as data_config_file:
        data_config = json.load(data_config_file)

    optimizer = getattr(optim, optimizer_type)(model.parameters(), lr=lr)
    criterion = getattr(nn, criterion_type)() #Default for BCELogitsLoss is mean reduction over the batch in question

    model = model.to(device)
    training_losses = []
    validation_losses = []
    epochs = []

    best_epoch = 0
    best_val_loss = np.inf
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
            # last_outputs, last_labels, last_flooded = outputs, labels, flooded

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
            if validation_epoch_average_loss < best_val_loss:
                best_epoch = epoch

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {validation_epoch_average_loss:.4f}')

        # Save model snapshot
        if epoch % 1000 == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{model.name}_{epoch}.pt"), hyperparams)
    
    # Save end model
    print("Best epoch:", best_epoch, "; Lowest validation loss:", best_val_loss)
    if is_final:
        save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{model.name}_{epoch}_FINAL.pt"), hyperparams)
    else:
        save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{model.name}_{epoch}.pt"), hyperparams)

    # PLOT EXAMPLE IMAGES ON VALIDATION
    # Select 8 samples
    if plot_training_images:
        if val_dataloader:
            model.eval()
            with torch.no_grad():
                size = 4
                flooded_images = []
                non_flooded_images = []
                for inputs, targets, flooded in val_dataloader:
                    inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)
                    # # Sort the tensors according to the sorted indices
                    # _, sorted_indices = torch.sort(flooded) #Arranges so we have non-flooded followed by flooded
                    # inputs = inputs[sorted_indices]
                    # targets = targets[sorted_indices]
                    # flooded = flooded[sorted_indices]
                    outputs = model(inputs)

                    for i in range(len(flooded)):
                        if flooded[i] == 1 and len(flooded_images) < size:
                            flooded_images.append((outputs[i], targets[i], flooded[i]))
                        elif flooded[i] == 0 and len(non_flooded_images) < size:
                            non_flooded_images.append((outputs[i], targets[i], flooded[i]))
                        
                        # Stop if we've collected 4 images in each category
                        if len(flooded_images) >= size and len(non_flooded_images) >= size:
                            break

                    if len(flooded_images) >= size and len(non_flooded_images) >= size:
                        break

                    # selected_outputs = outputs[:8]
                    # selected_targets = targets[:8]
                    # selected_targets_flooded = flooded[:8]
                    # break
                selected_outputs = [img[0] for img in flooded_images + non_flooded_images]
                selected_targets = [img[1] for img in flooded_images + non_flooded_images]
                selected_targets_flooded = [img[2] for img in flooded_images + non_flooded_images]
                image_examples_filename = os.path.join(data_config["validation_plots_path"], f"outputs_vs_labels_{model.name}.png")
                plot_model_output_vs_label_square(selected_outputs, selected_targets, selected_targets_flooded, image_examples_filename)
                print("Validation chart image saved!")
    
    # PLOT LOSS CHART
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
    model_to_save = model.module if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model
    torch.save({
        'model_type': model_to_save.__class__.__name__,
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparams': hyperparams
    }, filepath)
