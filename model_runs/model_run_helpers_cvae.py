import torch.optim as optim
from typing import Literal
from torch.utils.data import DataLoader
from dataloaders.convLSTM_dataset import *
from model_runs.model_run_helpers import *
from models.ConvLSTMSeparateBranches import *
from models.ConvLSTMMerged import *
from visualisations.visualisation_helpers import *


# Loss function: Reconstruction loss + KL Divergence
def KLReconstruction_loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (binary cross-entropy)
    BCE = torch.nn.BCELoss(recon_x, x, reduction='sum')
    
    # KL Divergence: D_KL(Q(z|x) || P(z))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + beta*KLD


def train_model_vae(data_config_path: str, model: torch.nn.Module, optimizer_type: str, lr: float, num_epochs: int, criterion_beta:int, device: torch.device, 
                    plot_training_images: bool, plot_losses: bool, 
                    train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader = None, 
                    is_final: bool = False) -> tuple:
    """
    Train a given model with specified hyperparameters and optionally validate, plot results, and save model checkpoints.

    Args:
        data_config_path (str): Path to the configuration file containing necessary file paths for saving models, plots, etc.
        model (torch.nn.Module): The model to be trained.
        optimizer_type (str): The optimizer type (e.g., 'Adam', 'SGD').
        lr (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): The device to train the model on (e.g., CPU or CUDA).
        plot_training_images (bool): If True, plots selected validation images during training.
        plot_losses (bool): If True, plots the training and validation loss curve.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset. Default is None.
        is_final (bool, optional): Whether this is the final model training (for naming the saved model). Default is False.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The trained model.
            - epoch (int): The final epoch number reached.

    """

    hyperparams = {
        'epochs': num_epochs,
        'batchsize': train_dataloader.batch_size,
        'lr': lr,
        'precedingrainfall': model.preceding_rainfall_days,
        'dropout': model.dropout_prob,
        'latentdims': model.latent_dims,
        'optim': optimizer_type,
        'criterionbeta': criterion_beta,
        'transforms': True if train_dataloader.dataset.transform else False,
        'res': train_dataloader.dataset.resolution
    }

    with open(data_config_path) as data_config_file:
        data_config = json.load(data_config_file)

    optimizer = getattr(optim, optimizer_type)(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=0.0001, threshold=0.0001, threshold_mode='rel')
    # criterion = getattr(nn, criterion_type)() #Default for BCELogitsLoss is mean reduction over the batch in question

    model = model.to(device)
    training_losses = []
    validation_losses = []
    epochs = []

    best_epoch = 0
    best_val_loss = np.inf
    early_stopping_patience = 75  # Stop training if no improvement after 50 epochs
    epochs_no_improve = 0
    for epoch in range(1, num_epochs+1):
        # TRAINING
        model.train()
        training_epoch_loss = 0.0
        num_batches = 0
        for inputs, labels, flooded in train_dataloader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            reconstructed, mu, logvar = model(labels, inputs)            
            loss = KLReconstruction_loss_function(reconstructed, labels, mu, logvar, criterion_beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # last_outputs, last_labels = torch.sigmoid(outputs), labels # apply sigmoid for charting purposes

            training_epoch_loss += loss.item()
            num_batches += 1

        # Save values down for loss chart plotting
        training_epoch_average_loss = training_epoch_loss/num_batches
        training_losses.append(training_epoch_average_loss)
        epochs.append(epoch)


        # COLLECT VALIDATION LOSSES
        if val_dataloader: # Check if we even want validation losses
            validation_epoch_average_loss = validate_model_vae(model, val_dataloader, criterion_beta, device)
            scheduler.step(validation_epoch_average_loss)
            validation_losses.append(validation_epoch_average_loss)
            if validation_epoch_average_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = validation_epoch_average_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stopping_patience:
                print("Final LR:", optimizer.param_groups[0]['lr'])
                save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{model.name}_{epoch}_earlystop.pt"), hyperparams)
                print(f'Early stopping triggered after {epoch} epochs')
                print("Best epoch:", best_epoch, "; Lowest validation loss:", best_val_loss)
                break

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {validation_epoch_average_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}")

        # Save model snapshot
        if epoch % 1000 == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{model.name}_{epoch}.pt"), hyperparams)
    
    # Save end model
    print("Final LR:", optimizer.param_groups[0]['lr'])
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
                    reconstructed, mu, logvar = model(targets, inputs)

                    for i in range(len(flooded)):
                        if flooded[i] == 1 and len(flooded_images) < size:
                            flooded_images.append((reconstructed[i], targets[i], flooded[i]))
                        elif flooded[i] == 0 and len(non_flooded_images) < size:
                            non_flooded_images.append((reconstructed[i], targets[i], flooded[i]))
                        
                        # Stop if we've collected 4 images in each category
                        if len(flooded_images) >= size and len(non_flooded_images) >= size:
                            break

                    if len(flooded_images) >= size and len(non_flooded_images) >= size:
                        break

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


def validate_model_vae(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion_beta: int, device: torch.device) -> float:
    """
    Validate the model on a given dataset by computing the average loss over the validation dataset.

    Args:
        model (torch.nn.Module): The model to be validated.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the validation dataset.
        criterion (torch.nn.Module): The loss function used to compute the loss between the predicted and target outputs.
        device (torch.device): The device (CPU or CUDA) on which to perform the validation.

    Returns:
        float: The average validation loss over the entire dataset.

    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets, flooded in dataloader:
            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)
            reconstructed, mu, logvar = model(targets, inputs)
            loss = KLReconstruction_loss_function(reconstructed, targets, mu, logvar, criterion_beta)
            total_loss += loss.item()
    return total_loss / len(dataloader)
