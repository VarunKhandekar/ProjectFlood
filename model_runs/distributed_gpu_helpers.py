import torch
import torch.optim as optim
import csv
import datetime
from typing import Literal
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# import optuna
# from optuna.integration import PyTorchLightningPruningCallback
from dataloaders.convLSTM_dataset import *
from models.ConvLSTMSeparateBranches import *
from final_evaluation.model_evaluation_helpers import *
from visualisations.visualisation_helpers import *
from model_runs.model_run_helpers import *


def get_attribute(model, attr):
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return getattr(model.module, attr)
    return getattr(model, attr)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()


def train_model_dist(rank: int, world_size: int, data_config_path: str, model,  criterion_type: str, optimizer_type: str, lr, num_epochs: int, 
                     plot_training_images: bool, plot_losses: bool, train_batch_size: int, 
                     train_dataset: Dataset, val_dataloader: DataLoader = None, is_final: bool = False):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])


    with open(data_config_path) as data_config_file:
        data_config = json.load(data_config_file)

    optimizer = getattr(optim, optimizer_type)(model.parameters(), lr=lr)
    criterion = getattr(nn, criterion_type)() #Default for BCE is mean reduction over the batch in question

    # Set up training dataloader
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, sampler=sampler)
    

    hyperparams = {
        'num_epochs': num_epochs,
        'train_batch_size': train_dataloader.batch_size,
        'learning_rate': lr,
        'preceding_rainfall_days': get_attribute(model, 'preceding_rainfall_days'),
        'dropout_prob': get_attribute(model, 'dropout_prob'),
        'output_channels': get_attribute(model, 'output_channels'),
        'conv_block_layers': get_attribute(model, 'conv_block_layers'),
        'convLSTM_layers': get_attribute(model, 'convLSTM_layers'),
        'optimizer_type': optimizer_type,
        'criterion': criterion_type  
    }
    # print(hyperparams)

    training_losses = []
    validation_losses = []
    epochs = []
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
            # last_outputs, last_labels, last_flooded = outputs, labels, flooded

            training_epoch_loss += loss.item()
            num_batches += 1

        # Save values down for loss chart plotting
        training_epoch_average_loss = training_epoch_loss/num_batches
        training_losses.append(training_epoch_average_loss)
        epochs.append(epoch)


        # COLLECT VALIDATION LOSSES
        if val_dataloader: # Check if we even can do validation losses
            validation_epoch_average_loss = validate_model(model, val_dataloader, criterion, rank)
            validation_losses.append(validation_epoch_average_loss)

        if epoch % 100 == 0 and rank == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {validation_epoch_average_loss:.4f}')

        # Save model snapshot
        if epoch % 500 == 0 and rank == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{get_attribute(model, 'name')}_{epoch}.pt"), hyperparams)
    
    # Save end model
    if rank == 0:
        if is_final:
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