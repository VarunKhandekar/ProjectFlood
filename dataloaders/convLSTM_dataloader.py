import imageio
import torch
import os
from torch.utils.data import Dataset
from data_extraction.generic_helpers import *
from dataloaders.dataloader_helpers import *
from dataloaders.custom_image_transforms import *

class FloodPredictionDataset(Dataset):
    def __init__(self, config_file_path, label_file_name, resolution, preceding_rainfall_days, forecast_rainfall_days, transform=None):
        with open(config_file_path) as config_file:
            config = json.load(config_file)
            
        self.rainfall_dir = f"{config['rainfall_path']}_{resolution}_{resolution}"
        self.topology_dir = f"{config['topology_path']}_{resolution}_{resolution}"
        self.soil_moisture_combo_dir = f"{config['soil_moisture_combo_path']}_{resolution}_{resolution}"
        self.water_images_dir = f"{config[label_file_name]}_{resolution}_{resolution}"
        self.preceding_rainfall_days = preceding_rainfall_days
        self.forecast_rainfall_days = forecast_rainfall_days
        self.transform = transform
        self.resolution = resolution

        self.rainfall_min, self.rainfall_max = get_log_rainfall_stats_training(f"{config['training_labels_path']}_{self.resolution}_{self.resolution}", 
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
        if self.transform:
            images, label = self.transform(images, label)
        
        image_tensor = torch.stack(torch.stack(images['preceding']), 
                                   torch.stack(images['forecast']), 
                                   images['topology'], 
                                   images['soil_moisture'])
        
        label_tensor = label.toTensor()
        
        return image_tensor, label_tensor
    