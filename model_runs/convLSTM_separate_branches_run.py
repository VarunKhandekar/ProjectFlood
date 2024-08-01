import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import date
from torch.utils.data import DataLoader
from dataloaders.convLSTM_dataloader import *
from dataloaders.custom_image_transforms import *
from models.convLSTM_separate_branches import *


# CONFIGURATION AND HYPERPARAMETERS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {0}'.format(device))
hyperparams = {
    'learning_rate': 0.01,
    'train_batch_size': 32,
    'optimizer_type': 'Adam',  # Note: storing the name as string for serialization
    'num_epochs': 100,
    'preceding_rainfall_days': 1,  # Example for a simple model layer configuration
    'dropout_prob': 0.0,
    'criterion': 'BCEWithLogitsLoss'
}

num_epochs = 100
train_batch_size = 32
val_batch_size = 32
test_batch_size = 8
learning_rate = 0.01
preceding_rainfall_days = 1
dropout_prob=0.0
optimizer = optim.Adam
criterion = nn.BCEWithLogitsLoss()

# Build the model
num_class = 1
model = ConvLSTMCombinedModel(preceding_rainfall_days=preceding_rainfall_days, forecast_rainfall_days=1)
model = model.to(device)
params = list(model.parameters())

with open(os.environ["PROJECT_FLOOD_DATA"]) as config_file_path:
    config_data = json.load(config_file_path)

if not os.path.exists(config_data["saved_models_path"]):
    os.makedirs(config_data["saved_models_path"])


#Set up dataloaders
#Set up model
#Train model
#Evaluate model (maybe...)

# # Optimizer
# optimizer = optim.Adam(params, lr=learning_rate)

# # Segmentation loss
# criterion = nn.BCEWithLogitsLoss()

# Datasets
train_set = FloodPredictionDataset(os.environ["PROJECT_FLOOD_DATA"], "training_labels_path", resolution=256, preceding_rainfall_days=1, forecast_rainfall_days=1, transform=train_transform)
validation_set = FloodPredictionDataset(os.environ["PROJECT_FLOOD_DATA"], "validation_labels_path", resolution=256, preceding_rainfall_days=1, forecast_rainfall_days=1, transform=train_transform)
test_set = FloodPredictionDataset(os.environ["PROJECT_FLOOD_DATA"], "test_labels_path", resolution=256, preceding_rainfall_days=1, forecast_rainfall_days=1, transform=None)



train_dataloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=4)
validation_dataloader = DataLoader(validation_set, batch_size=val_batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=4)

# Train the model


start = time.time()

# Set the modules in training mode, which will have effects on certain modules, e.g. dropout or batchnorm.
model.train()
for epoch in range(1, 1 + num_epochs):
    start_epoch = time.time()
    for images, label in train_dataloader:
        images, label = images.to(device, dtype=torch.float32), label.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(images)
        # logits = torch.sigmoid(outputs)
        # print(outputs.shape, label.shape)
        loss = criterion(outputs, label) #pytorch does internal reshaping, BCEwithLogitsLoss means we don't need to apply sigmoid
        loss.backward()
        optimizer.step()
    print(f"Training loss, batch {epoch} :  {loss.item():.3f}")
    

torch.save(model.state_dict(), os.path.join(config_data["saved_models_path"], f"convLSTM_separate_branch_{epoch}_{date.today().strftime(r'%Y%m%d')}.pt"))