import os
import pandas as pd
import json
import random
import math
import shutil
from typing import Literal


def split_files(files, val_ratio, test_ratio):
    random.shuffle(files)
    
    total_files = len(files)
    train_end = math.floor(total_files * (1-test_ratio-val_ratio))
    val_end = train_end + math.floor(total_files * val_ratio)
    
    # Split the files
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    return train_files, val_files, test_files


def split_training_test(config_file, resolution, flood_dates_file, val_ratio, test_ratio):
    assert val_ratio + test_ratio <= 0.5, "Validation and test ratios combined should not exceed 0.5"  

    with open("static/config.json") as config_file:
        config = json.load(config_file)
    water_images_path = f"{config['water_image_path']}_{resolution}_{resolution}"

    flood_dates = pd.read_excel(flood_dates_file)
    flood_dates = flood_dates["Dates"].tolist()
    flood_files = [f"BangladeshWater{i.strftime(r'%Y%m%d')}.tif" for i in flood_dates]
    flood_files_set = set(flood_files)

    non_flood_files = [i for i in os.listdir(water_images_path) if i not in flood_files_set]

    # Split the files for flood
    train_files_flood, val_files_flood, test_files_flood = split_files(flood_files, val_ratio, test_ratio)

    # Split the files for non flood
    train_files_non_flood, val_files_non_flood, test_files_non_flood = split_files(non_flood_files, val_ratio, test_ratio)

    # Combine
    train_files = train_files_flood + train_files_non_flood
    val_files = val_files_flood + val_files_non_flood
    test_files = test_files_flood + test_files_non_flood

    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    return train_files, val_files, test_files

def save_files(config_file, resolution, files, dataset_type: Literal['training', 'validation', 'test']):
    dataset_types = ['training', 'validation', 'test']
    if dataset_type not in dataset_types:
        raise ValueError(f"Invalid value provided. Choose from {dataset_types}")
    
    with open(config_file) as config_file:
        config = json.load(config_file)
    
    if dataset_type == 'training':
        target_directory = f"{config['training_labels_path']}_{resolution}_{resolution}"
    
    elif dataset_type == 'validation':
        target_directory = f"{config['validation_labels_path']}_{resolution}_{resolution}"
    
    elif dataset_type == 'test':
        target_directory = f"{config['test_labels_path']}_{resolution}_{resolution}"

    # remove existing directory and create new one
    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)  # Removes the directory and all its contents
    os.mkdir(target_directory)

    for f in files:
        source_root = f"{config['water_image_path']}_{resolution}_{resolution}"
        source = os.path.join(source_root, f)
        destination = os.path.join(target_directory, f)
        shutil.copy(source, destination)


if __name__ == "__main__":
    random.seed(42)
    resolution = 256
    test_ratio = 0.2
    val_ratio = 0.2
    config_file = "static/config.json"
    train_files, val_files, test_files = split_training_test(config_file, 
                                                             resolution, 
                                                             "static/final_flood_dates.xlsx", 
                                                             val_ratio, 
                                                             test_ratio)
    
    save_files(config_file, resolution, train_files,'training')
    save_files(config_file, resolution, val_files, 'validation')
    save_files(config_file, resolution, test_files, 'test')
    