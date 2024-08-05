import torch
import torch.nn as nn
import torch.optim as optim
import os
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from dataloaders.convLSTM_dataset import *
from dataloaders.custom_image_transforms import *
from models.ConvLSTMSeparateBranches import *
from model_runs.model_run_helpers import *


# CONFIGURATION AND HYPERPARAMETERS
model_run_date = datetime.date.today().strftime(r'%Y%m%d')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {0}'.format(device))

with open(os.environ["PROJECT_FLOOD_DATA"]) as config_file_path:
    config_data = json.load(config_file_path)

if not os.path.exists(config_data["saved_models_path"]):
    os.makedirs(config_data["saved_models_path"])

resolution = 256

def objective(trial):
    num_epochs = trial.suggest_categorical('num_epochs', [i for i in range(1000, 5000+1, 1000)])
    train_batch_size = trial.suggest_categorical('train_batch_size', [16, 32, 64])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    preceding_rainfall_days = trial.suggest_categorical('train_batch_size', [i for i in range(1, 7+1)])
    dropout_prob = trial.suggest_uniform('dropout_prob', 0.0, 0.5)
    output_channels = trial.suggest_int('output_channels', 8, 32)
    conv_block_layers = trial.suggest_int('conv_block_layers', 1, 4)
    convLSTM_layers = trial.suggest_int('convLSTM_layers', 1, 3)
    optimizer_str = trial.suggest_categorical('optimizer_str', ['Adam', 'SGD'])
    criterion_str = 'BCEWithLogitsLoss'

    hyperparams = {
        'num_epochs': num_epochs,
        'trainbatchsize': train_batch_size,
        'lr': learning_rate,
        'precedingrainfalldays': preceding_rainfall_days,
        'dropoutprob': dropout_prob,
        'outputchannels': output_channels,
        'convblocklayers': conv_block_layers,
        'convLSTMlayers': convLSTM_layers,
        'optimizer': optimizer_str,
        'criterion': criterion_str
    }

    # Base Dataset
    train_val_dataset = FloodPredictionDataset(os.environ["PROJECT_FLOOD_DATA"], 'training_validation_combo_path', resolution, preceding_rainfall_days, 1, transform=None)

    kf = KFold(n_splits=5, shuffle=True, random_state=42) #random state initialised for consistency across function calls

    indices = list(range(len(train_val_dataset)))

    for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
        train_subset = Subset(train_val_dataset, train_indices)
        val_subset = Subset(train_val_dataset, val_indices)

        # Apply transformations
        train_dataset = SubsetWithTransform(train_subset, transform=train_transform) # ENSURING TRAINING HAS DATA AUGMENTATION
        val_dataset = SubsetWithTransform(val_subset, transform=None) # BUT VALIDATION DOES NOT

        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

        # Build the model
        model = ConvLSTMSeparateBranches(preceding_rainfall_days, 1, output_channels, conv_block_layers, convLSTM_layers, dropout_prob)

        model_name = generate_model_name(model.__class__.__name__, model_run_date, **hyperparams)
        model.name = model_name

        with open(os.environ["PROJECT_FLOOD_DATA"]) as data_config_file:
            data_config = json.load(data_config_file)

        optimizer = getattr(optim, optimizer_str)(model.parameters(), lr=learning_rate)
        criterion = getattr(nn, criterion_str)() #Default for BCELogitsLoss is mean reduction over the batch in question

        model = model.to(device)
        training_losses = []
        validation_losses = []
        epochs = []
        for epoch in range(1, num_epochs+1):
            # TRAINING
            model.train()
            training_epoch_loss = 0.0
            num_batches = 0
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # last_outputs, last_labels = outputs, labels
                last_outputs, last_labels = torch.sigmoid(outputs), labels # apply sigmoid for charting purposes

                training_epoch_loss += loss.item()
                num_batches += 1

            # Save values down for loss chart plotting
            training_epoch_average_loss = training_epoch_loss/num_batches
            training_losses.append(training_epoch_average_loss)
            epochs.append(epoch)


            # COLLECT VALIDATION LOSSES
            validation_epoch_average_loss = validate_model(model, val_dataloader, criterion, device)
            validation_losses.append(validation_epoch_average_loss)
            trial.report(validation_epoch_average_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # # Save model snapshot
            # if epoch % 1000 == 0:
            #     save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{model.name}_{epoch}.pt"), hyperparams)
        
        # Save final model            
        save_checkpoint(model, optimizer, epoch, os.path.join(data_config["saved_models_path"], f"{model.name}_{epoch}.pt"), hyperparams)

        # PLOT EXAMPLE IMAGES FROM TRAINING
        # Select only 4 samples from the last batch
        selected_outputs = last_outputs[:4]
        selected_labels = last_labels[:4]
        image_examples_filename = os.path.join(data_config["training_plots_path"], f"outputs_vs_labels_{model.name}.png")
        plot_model_output_vs_label(selected_outputs, selected_labels, image_examples_filename)
        
        # PLOT LOSS CHART
        losses = []
        losses.append(training_losses)
        if not validation_losses:
            losses.append(validation_losses)
        loss_filename = os.path.join(data_config["loss_plots_path"], f"losschart_{model.name}.png")
        plot_loss_chart(losses, epochs, loss_filename)

        return sum(validation_losses) / len(validation_losses)




# Set up and run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# Save the best hyperparameters
with open(f"ConvLSTMSeparateBranches_best_hyperparams_{model_run_date}.json", "w") as f:
    json.dump(study.best_trial.params, f)
