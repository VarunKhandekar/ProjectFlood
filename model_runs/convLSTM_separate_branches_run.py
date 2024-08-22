import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from dataloaders.convLSTM_dataset import *
from dataloaders.custom_image_transforms import *
from models.ConvLSTMSeparateBranches import *
from model_runs.model_run_helpers import *


# CONFIGURATION AND HYPERPARAMETERS
model_run_date = datetime.date.today().strftime(r'%Y%m%d')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {0}'.format(device))
torch.manual_seed(42)

num_epochs = 4000
train_batch_size = 32
learning_rate = 0.001
preceding_rainfall_days = 1
dropout_prob = 0.3
output_channels = 16
conv_block_layers = 2
convLSTM_layers = 1
optimizer_str = 'RMSprop'
criterion_str = 'BCELoss'
resolution = 256
transforms = False


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


# Evaluate the model


# accuracy, average_loss, metrics = evaluate_model(os.environ['PROJECT_FLOOD_DATA'], model, test_dataloader, criterion_str, device, end_epoch, model_run_date)
