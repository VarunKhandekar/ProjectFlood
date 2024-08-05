import imageio.v2 as imageio
import torch
import os
from torch.utils.data import Dataset
from data_extraction.generic_helpers import *
from dataloaders.dataloader_helpers import *
from dataloaders.custom_image_transforms import *

class FloodPredictionDataset(Dataset):
    def __init__(self, data_config_path, label_file_name, resolution, preceding_rainfall_days, forecast_rainfall_days, transform=None):
        with open(data_config_path) as data_config_file:
            data_config = json.load(data_config_file)
            
        self.rainfall_dir = f"{data_config['rainfall_path']}_{resolution}_{resolution}"
        self.topology_dir = f"{data_config['topology_path']}_{resolution}_{resolution}"
        self.soil_moisture_combo_dir = f"{data_config['soil_moisture_combo_path']}_{resolution}_{resolution}"
        self.water_images_dir = f"{data_config[label_file_name]}_{resolution}_{resolution}"
        self.preceding_rainfall_days = preceding_rainfall_days
        self.forecast_rainfall_days = forecast_rainfall_days
        self.transform = transform
        self.resolution = resolution

        self.rainfall_min, self.rainfall_max = get_log_rainfall_stats_training(f"{data_config['training_labels_path']}_{self.resolution}_{self.resolution}", 
                                                                               self.rainfall_dir,
                                                                               self.preceding_rainfall_days)
        
    def __len__(self):
        return len(os.listdir(self.water_images_dir))

    def __getitem__(self, idx):

        label_name = sorted(os.listdir(self.water_images_dir))[idx]
        label = imageio.imread(os.path.join(self.water_images_dir, label_name))

        # Get images, transform each if needed, then combine into a single tensor
        images = generate_label_images(label_name, 
                                       self.soil_moisture_combo_dir, 
                                       self.topology_dir, 
                                       self.rainfall_dir, 
                                       self.rainfall_min, 
                                       self.rainfall_max, 
                                       self.preceding_rainfall_days)
        
        # Convert images to tensors
        label_tensor = torch.tensor(label)
        image_tensors = {}
        for key, value in images.items():
            tensor_images = [torch.tensor(image, dtype=torch.float32) for image in value]
            image_tensors[key] = tensor_images
        
        if self.transform:
            image_tensors, label_tensor = self.transform(image_tensors, label_tensor)
        
        preceding_stacked = prepare_tensors(image_tensors['preceding'])
        forecast_stacked = prepare_tensors(image_tensors['forecast'])
        soil_moisture_stacked = prepare_tensors(image_tensors['soil_moisture'])
        topology_stacked = prepare_tensors(image_tensors['topology'])

        image_tensor = torch.cat([preceding_stacked, forecast_stacked, topology_stacked, soil_moisture_stacked], dim=0)
        
        return image_tensor, label_tensor
    

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image_tensors, label_tensor = self.subset[idx]
        if self.transform:
            image_tensors, label_tensor = self.transform(image_tensors, label_tensor)
        return image_tensors, label_tensor