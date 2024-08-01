import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from dataloaders.convLSTM_dataloader import *
from dataloaders.custom_image_transforms import *
from models.convLSTM_separate_branches import *
from model_runs.model_run_helpers import *


# CONFIGURATION AND HYPERPARAMETERS
model_run_date = date.today().strftime(r'%Y%m%d')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {0}'.format(device))
# hyperparams = {
#     'learning_rate': 0.01,
#     'train_batch_size': 32,
#     'optimizer_type': 'Adam',  # Note: storing the name as string for serialization
#     'num_epochs': 100,
#     'preceding_rainfall_days': 1,  # Example for a simple model layer configuration
#     'dropout_prob': 0.0,
#     'criterion': 'BCEWithLogitsLoss'
# }

num_epochs = 10000
train_batch_size = 32
learning_rate = 0.01
preceding_rainfall_days = 1
dropout_prob=0.0
optimizer_str = 'Adam'
criterion_str = 'BCEWithLogitsLoss'

with open(os.environ["PROJECT_FLOOD_DATA"]) as config_file_path:
    config_data = json.load(config_file_path)

if not os.path.exists(config_data["saved_models_path"]):
    os.makedirs(config_data["saved_models_path"])

# Build the model
model = ConvLSTMSeparateBranches(preceding_rainfall_days, 1, dropout_prob)
model = model.to(device)
params = list(model.parameters())

# optimizer = getattr(optim, hyperparams['optimizer_type'])(model.parameters(), lr=hyperparams['learning_rate'])

optimizer = optim.Adam(params, lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()


# Train the model
train_dataloader = get_dataloader("training_labels_path", resolution=256, preceding_rainfall_days=preceding_rainfall_days, forecast_rainfall_days=1, 
                                  transform=train_transform, batch_size=train_batch_size, shuffle=True, num_workers=4)
model, end_epoch = train_model(os.environ['PROJECT_FLOOD_DATA'], model, train_dataloader, criterion, optimizer, learning_rate, num_epochs, device, model_run_date)


# Evaluate the model
test_batch_size = 8
test_dataloader = get_dataloader("test_labels_path", resolution=256, preceding_rainfall_days=preceding_rainfall_days, forecast_rainfall_days=1, 
                                  transform=train_transform, batch_size=train_batch_size, shuffle=True, num_workers=4)

accuracy, average_loss, metrics = evaluate_model(os.environ['PROJECT_FLOOD_DATA'], model, test_dataloader, criterion, device, end_epoch, model_run_date)
