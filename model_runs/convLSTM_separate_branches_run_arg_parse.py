import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import time
from torch.utils.data import DataLoader
from dataloaders.convLSTM_dataset import *
from dataloaders.custom_image_transforms import *
from models.ConvLSTMSeparateBranches import *
from model_runs.model_run_helpers import *

if __name__ == "__main__":
    # CONFIGURATION AND HYPERPARAMETERS
    start_time = time.time()
    model_run_date = datetime.date.today().strftime(r'%Y%m%d')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {0}'.format(device))
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description='Training parameters')

    parser.add_argument('--num_epochs', type=int, default=8000)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--preceding_rainfall_days', type=int, default=1)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--output_channels', type=int, default=16)
    parser.add_argument('--conv_block_layers', type=int, default=4)
    parser.add_argument('--convLSTM_layers', type=int, default=2)
    parser.add_argument('--optimizer_str', type=str, default='RMSprop')
    parser.add_argument('--criterion_str', type=str, default='BCELoss')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--transforms', type=str, default='False')

    args = parser.parse_args()

    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    learning_rate = args.learning_rate
    preceding_rainfall_days = args.preceding_rainfall_days
    dropout_prob = args.dropout_prob
    output_channels = args.output_channels
    conv_block_layers = args.conv_block_layers
    convLSTM_layers = args.convLSTM_layers
    optimizer_str = args.optimizer_str
    criterion_str = args.criterion_str
    resolution = args.resolution
    transforms = True if args.transforms.lower() == 'true' else False


    hyperparams = {
        'epochs': num_epochs,
        'batchsize': train_batch_size,
        'lr': learning_rate,
        'precedingrainfall': preceding_rainfall_days,
        'dropout': dropout_prob,
        'outputchannels': output_channels,
        'convblocklayers': conv_block_layers,
        'convLSTMlayers': convLSTM_layers,
        'optim': optimizer_str,
        'criterion': criterion_str,
        'transforms': transforms,
        'res': resolution
    }
    print(hyperparams)



    with open(os.environ["PROJECT_FLOOD_DATA"]) as config_file_path:
        config_data = json.load(config_file_path)

    if not os.path.exists(config_data["saved_models_path"]):
        os.makedirs(config_data["saved_models_path"])

    # Build the model
    model = ConvLSTMSeparateBranches(preceding_rainfall_days, 1, output_channels, conv_block_layers, convLSTM_layers, dropout_prob)
    model = model.to(device)
    # params = list(model.parameters())

    model_name = generate_model_name(model.__class__.__name__, model_run_date, **hyperparams)
    model.name = model_name

    # optimizer = getattr(optim, hyperparams['optimizer_type'])(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(params, lr=learning_rate)
    # criterion = nn.BCEWithLogitsLoss()

    # Set up dataloaders
    validation_batch_size = 16
    test_batch_size = 1

    if transforms:
        train_dataloader = get_dataloader("training_labels_path", resolution=resolution, preceding_rainfall_days=preceding_rainfall_days, forecast_rainfall_days=1, 
                                        transform=train_transform, batch_size=train_batch_size, shuffle=True, num_workers=4)
    else:
        train_dataloader = get_dataloader("training_labels_path", resolution=resolution, preceding_rainfall_days=preceding_rainfall_days, forecast_rainfall_days=1, 
                                        transform=None, batch_size=train_batch_size, shuffle=True, num_workers=4)
        
    val_dataloader = get_dataloader("validation_labels_path", resolution=resolution, preceding_rainfall_days=preceding_rainfall_days, forecast_rainfall_days=1, 
                                    transform=None, batch_size=validation_batch_size, shuffle=False, num_workers=4)
    test_dataloader = get_dataloader("test_labels_path", resolution=resolution, preceding_rainfall_days=preceding_rainfall_days, forecast_rainfall_days=1, 
                                    transform=None, batch_size=test_batch_size, shuffle=False, num_workers=4)


    # Train the model
    model, end_epoch = train_model(os.environ['PROJECT_FLOOD_DATA'], model, criterion_str, optimizer_str, learning_rate, num_epochs, device, True, True, train_dataloader, val_dataloader)

    end_time = time.time()
    elapsed_time = end_time - start_time
    days = elapsed_time // (24 * 3600)
    hours = (elapsed_time % (24 * 3600)) // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60
    print(f"{int(days)}-{int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    print("\n\n")
    