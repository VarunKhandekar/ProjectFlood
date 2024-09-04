import torch.distributed as dist
from typing import Any
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloaders.convLSTM_dataset import *
from models.ConvLSTMSeparateBranches import *
from visualisations.visualisation_helpers import *
from model_runs.model_run_helpers import *


def get_attribute(model: torch.nn.Module, attr: str) -> Any:
    """
    Retrieve an attribute from a model, handling both regular and distributed (DataParallel/DistributedDataParallel) models.

    Args:
        model (torch.nn.Module): The model from which to retrieve the attribute.
        attr (str): The name of the attribute to retrieve.

    Returns:
        Any: The value of the requested attribute from the model.

    """
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return getattr(model.module, attr)
    return getattr(model, attr)


def setup(rank: int, world_size: int) -> None:
    """
    Set up the environment for distributed training by initializing the process group.

    Args:
        rank (int): The rank of the current process in the distributed setup.
        world_size (int): The total number of processes participating in the distributed training.

    Returns:
        None: The function sets up the distributed environment for training.

    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42)


def cleanup() -> None:
    """
    Clean up the distributed process group by destroying it after distributed training is complete.

    Args:
        None

    Returns:
        None: The function destroys the process group, cleaning up resources used for distributed training.

    """
    dist.destroy_process_group()


def train_model_dist(rank: int, world_size: int, data_config_path: str, model: torch.nn.Module, criterion_type: str, optimizer_type: str, lr: float, num_epochs: int, 
                     plot_training_images: bool, plot_losses: bool, train_batch_size: int, 
                     train_dataset: torch.utils.data.Dataset, val_dataloader: torch.utils.data.DataLoader = None, is_final: bool = False) -> tuple:
    """
    Train a model using distributed data parallelism (DDP) across multiple GPUs with specified hyperparameters. Optionally validate, plot results, and save model checkpoints.

    Args:
        rank (int): The rank of the current process in the distributed setup.
        world_size (int): The total number of processes participating in the distributed training.
        data_config_path (str): Path to the configuration file containing necessary file paths for saving models, plots, etc.
        model (torch.nn.Module): The model to be trained.
        criterion_type (str): The loss function to be used (e.g., 'BCEWithLogitsLoss', 'MSELoss').
        optimizer_type (str): The optimizer type (e.g., 'Adam', 'SGD').
        lr (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train the model.
        plot_training_images (bool): If True, plots selected validation images during training.
        plot_losses (bool): If True, plots the training and validation loss curve.
        train_batch_size (int): The batch size for the training DataLoader.
        train_dataset (torch.utils.data.Dataset): The dataset for training the model.
        val_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset. Default is None.
        is_final (bool, optional): Whether this is the final model training (for naming the saved model). Default is False.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The trained model.
            - epoch (int): The final epoch number reached.

    Notes:
        This function uses Distributed Data Parallel (DDP) for multi-GPU training. It sets up a distributed environment, synchronizes training across processes, and handles validation and early stopping in a distributed manner.
    """
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])


    with open(data_config_path) as data_config_file:
        data_config = json.load(data_config_file)

    optimizer = getattr(optim, optimizer_type)(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=0.0001, threshold=0.0001, threshold_mode='rel')
    criterion = getattr(nn, criterion_type)() #Default for BCE is mean reduction over the batch in question

    # Set up training dataloader
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, sampler=sampler)

    hyperparams = {
        'epochs': num_epochs,
        'batchsize': train_dataloader.batch_size,
        'lr': lr,
        'precedingrainfall': get_attribute(model, 'preceding_rainfall_days'),
        'dropout': get_attribute(model, 'dropout_prob'),
        'outputchannels': get_attribute(model, 'output_channels'),
        'convblocklayers': get_attribute(model, 'conv_block_layers'),
        'convLSTMlayers': get_attribute(model, 'convLSTM_layers'),
        'optim': optimizer_type,
        'criterion': criterion_type,
        'transforms': True if train_dataset.transform else False,
        'res': train_dataset.resolution
    }

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
            inputs, labels = inputs.to(rank, dtype=torch.float32), labels.to(rank, dtype=torch.float32)
            outputs = model(inputs)            
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # last_outputs, last_labels = torch.sigmoid(outputs), labels # apply sigmoid for charting purposes
            training_epoch_loss += loss.item()
            num_batches += 1

        # Save values down for loss chart plotting
        # Synchronise across GPUs
        training_epoch_loss_tensor = torch.tensor(training_epoch_loss, dtype=torch.float, device='cuda')
        dist.all_reduce(training_epoch_loss_tensor, op=dist.ReduceOp.SUM)
        training_epoch_average_loss = training_epoch_loss_tensor.item() / (num_batches * dist.get_world_size())
        if rank == 0: #Only bother collecting in main process
            training_losses.append(training_epoch_average_loss)
            epochs.append(epoch)

        # COLLECT VALIDATION LOSSES; get best epoch 
        if val_dataloader: # Check if we even can do validation losses
            validation_epoch_average_loss = validate_model(model, val_dataloader, criterion, rank)

            #Synchronise across GPUs; get average validation_epoch loss
            validation_epoch_average_loss_tensor = torch.tensor([validation_epoch_average_loss], dtype=torch.float, device='cuda')
            dist.all_reduce(validation_epoch_average_loss_tensor, op=dist.ReduceOp.SUM)
            validation_epoch_average_loss = validation_epoch_average_loss_tensor.item() / dist.get_world_size()

            scheduler.step(validation_epoch_average_loss) #adjust LR based on that
            validation_losses.append(validation_epoch_average_loss)

            is_best = validation_epoch_average_loss < best_val_loss
            #Broadcast if this is the best average val loss seen so far. Set this for all GPUs
            is_best_tensor = torch.tensor([is_best], dtype=torch.int, device='cuda')
            dist.broadcast(is_best_tensor, src=0) #Any other GPUs will be forced to wait until rank=0 catches up if it is behind
            is_best = is_best_tensor.item()
            
            if is_best:
                best_epoch = epoch
                best_val_loss = validation_epoch_average_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            #Broadcast these rank=0 variables to all GPUs
            best_val_loss_tensor = torch.tensor([best_val_loss], dtype=torch.float, device='cuda')
            epochs_no_improve_tensor = torch.tensor([epochs_no_improve], dtype=torch.int, device='cuda')
            dist.broadcast(best_val_loss_tensor, src=0)
            dist.broadcast(epochs_no_improve_tensor, src=0)
            best_val_loss = best_val_loss_tensor.item()
            epochs_no_improve = epochs_no_improve_tensor.item()
            
            if epochs_no_improve >= early_stopping_patience:
                if rank == 0: #Only print main process bits
                    save_checkpoint(model, optimizer, best_epoch, os.path.join(data_config["saved_models_path"], f"{get_attribute(model, 'name')}_{epoch}_earlystop.pt"), hyperparams)
                    print(f'Early stopping triggered after {epoch} epochs')
                    print("Best epoch:", best_epoch, "; Lowest validation loss:", best_val_loss)
                    print("Final LR:", optimizer.param_groups[0]['lr'])
                    break

            if epoch % 100 == 0 and rank == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {validation_epoch_average_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}")

        # Save model snapshot
        if epoch % 500 == 0 and rank == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{get_attribute(model, 'name')}_{epoch}.pt"), hyperparams)
    
    # Save end model
    if rank == 0:
        if is_final: #Early stopping epoch is saved
            save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{get_attribute(model, 'name')}_{epoch}_FINAL.pt"), hyperparams)
        else:
            save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{get_attribute(model, 'name')}_{epoch}.pt"), hyperparams)

    # PLOT EXAMPLE IMAGES ON VALIDATION
    # Select 8 samples from validation batch
    if rank == 0:
        if plot_training_images:
            if val_dataloader:
                model.eval()
                with torch.no_grad():
                    size = 4
                    flooded_images = []
                    non_flooded_images = []
                
                    for inputs, targets, flooded in val_dataloader:
                        inputs, targets = inputs.to(rank, dtype=torch.float32), targets.to(rank, dtype=torch.float32)
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

                    selected_outputs = [img[0] for img in flooded_images + non_flooded_images]
                    selected_targets = [img[1] for img in flooded_images + non_flooded_images]
                    selected_targets_flooded = [img[2] for img in flooded_images + non_flooded_images]

                    image_examples_filename = os.path.join(data_config["validation_plots_path"], f"outputs_vs_labels_{get_attribute(model, 'name')}.png")
                    plot_model_output_vs_label_square(selected_outputs, selected_targets, selected_targets_flooded, image_examples_filename)
                    print("Validation chart image saved!")
        
        # PLOT LOSS CHART
        if plot_losses:
            losses = []
            losses.append(training_losses)
            if validation_losses:
                losses.append(validation_losses)
            loss_filename = os.path.join(data_config["loss_plots_path"], f"losschart_{get_attribute(model, 'name')}.png")
            plot_loss_chart(losses, epochs, loss_filename, hyperparams)
            print("Loss chart image saved!")

    cleanup()
    return model, epoch