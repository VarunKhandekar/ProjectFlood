import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from dataloaders.convLSTM_dataset import *
from dataloaders.custom_image_transforms import *
from models.ConvLSTMSeparateBranches import *
from model_runs.model_run_helpers import *
from model_runs.distributed_gpu_helpers import *



if __name__ == "__main__":
    # CONFIGURATION AND HYPERPARAMETERS
    model_run_date = datetime.date.today().strftime(r'%Y%m%d')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Device: {0}'.format(device))
    # print(torch.cuda.device_count())
    world_size = torch.cuda.device_count()
    torch.manual_seed(42)

    num_epochs = 3000
    train_batch_size = 128
    learning_rate = 0.0001
    preceding_rainfall_days = 1
    dropout_prob = 0.0
    output_channels = 16
    conv_block_layers = 4
    convLSTM_layers = 2
    optimizer_str = 'RMSprop'
    criterion_str = 'BCELoss'


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


    with open(os.environ["PROJECT_FLOOD_DATA"]) as config_file_path:
        config_data = json.load(config_file_path)

    if not os.path.exists(config_data["saved_models_path"]):
        os.makedirs(config_data["saved_models_path"])

    # Build the model
    model = ConvLSTMSeparateBranches(preceding_rainfall_days, 1, output_channels, conv_block_layers, convLSTM_layers, dropout_prob)
    model_name = generate_model_name(model.__class__.__name__, model_run_date, **hyperparams)
    model.name = model_name


    # Specifics for training
    train_dataset = FloodPredictionDataset(os.environ["PROJECT_FLOOD_DATA"], os.environ["PROJECT_FLOOD_CORE_PATHS"], 
                                           "training_labels_path", resolution=256, preceding_rainfall_days=preceding_rainfall_days, 
                                            forecast_rainfall_days=1, transform=None)

    # Set up dataloaders
    validation_batch_size = 16
    test_batch_size = 1

    val_dataloader = get_dataloader("validation_labels_path", resolution=256, preceding_rainfall_days=preceding_rainfall_days, forecast_rainfall_days=1, 
                                    transform=None, batch_size=validation_batch_size, shuffle=False, num_workers=4)
    test_dataloader = get_dataloader("test_labels_path", resolution=256, preceding_rainfall_days=preceding_rainfall_days, forecast_rainfall_days=1, 
                                    transform=None, batch_size=test_batch_size, shuffle=False, num_workers=4)
    
    # Train the model
    torch.multiprocessing.spawn(train_model_dist, args=(world_size, os.environ['PROJECT_FLOOD_DATA'], 
                                                        model, 
                                                        criterion_str, 
                                                        optimizer_str, 
                                                        learning_rate, 
                                                        num_epochs,
                                                        True,
                                                        True,
                                                        train_batch_size,
                                                        train_dataset,
                                                        val_dataloader), nprocs=world_size)
