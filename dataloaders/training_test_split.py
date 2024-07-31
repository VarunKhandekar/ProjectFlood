import os
import pandas as pd
import json
import random
import math
import shutil
from typing import Literal


def split_files(files: list, val_ratio: float, test_ratio: float) -> tuple:
    """
    Split a list of files into training, validation, and test sets.

    Args:
        files (list): List of files to be split.
        val_ratio (float): Ratio of files to be used for validation.
        test_ratio (float): Ratio of files to be used for testing.

    Returns:
        tuple: Three lists representing the training, validation, and test sets.
    """
    random.shuffle(files)
    
    total_files = len(files)
    train_end = math.floor(total_files * (1-test_ratio-val_ratio))
    val_end = train_end + math.floor(total_files * val_ratio)
    
    # Split the files
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    return train_files, val_files, test_files


def split_training_test(data_config_path: str, resolution: int, flood_dates_file: str, val_ratio: float, test_ratio: float) -> tuple:
    """
    Split files into training, validation, and test sets based on flood and non-flood dates.

    Args:
        config_file (str): Path to the configuration file.
        resolution (int): Desired resolution of the images.
        flood_dates_file (str): Path to the file containing flood dates.
        val_ratio (float): Ratio of files to be used for validation.
        test_ratio (float): Ratio of files to be used for testing.

    Returns:
        tuple: Three lists representing the training, validation, and test sets.
    """
    assert val_ratio + test_ratio <= 0.5, "Validation and test ratios combined should not exceed 0.5"  

    with open(data_config_path) as data_config_file:
        data_config = json.load(data_config_file)
    water_images_path = f"{data_config['water_image_path']}_{resolution}_{resolution}"

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

def save_files(data_config_path: str, resolution: int, files: list, dataset_type: Literal['training', 'validation', 'test']):
    """
    Save files to their respective directories for training, validation, or testing.

    Args:
        config_file (str): Path to the configuration file.
        resolution (int): Desired resolution of the images.
        files (list): List of files to be saved.
        dataset_type (str): Type of dataset to save the files to. Must be one of 'training', 'validation', or 'test'.

    Raises:
        ValueError: If an invalid dataset type is provided.
    """
    dataset_types = ['training', 'validation', 'test']
    if dataset_type not in dataset_types:
        raise ValueError(f"Invalid value provided. Choose from {dataset_types}")
    
    with open(data_config_path) as data_config_file:
        data_config = json.load(data_config_file)
    
    if dataset_type == 'training':
        target_directory = f"{data_config['training_labels_path']}_{resolution}_{resolution}"
    
    elif dataset_type == 'validation':
        target_directory = f"{data_config['validation_labels_path']}_{resolution}_{resolution}"
    
    elif dataset_type == 'test':
        target_directory = f"{data_config['test_labels_path']}_{resolution}_{resolution}"

    # remove existing directory and create new one
    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)  # Removes the directory and all its contents
    os.mkdir(target_directory)

    for f in files:
        source_root = f"{data_config['water_image_path']}_{resolution}_{resolution}"
        source = os.path.join(source_root, f)
        destination = os.path.join(target_directory, f)
        shutil.copy(source, destination)


if __name__ == "__main__":
    # 60-20-20 SPLIT
    random.seed(42)
    resolution = 256
    test_ratio = 0.2
    val_ratio = 0.2
    data_config_file = os.environ["PROJECT_FLOOD_DATA"]
    with open(os.environ["PROJECT_FLOOD_CORE_PATHS"]) as core_config_file:
        core_config = json.load(core_config_file)
    train_files, val_files, test_files = split_training_test(data_config_file, 
                                                             resolution, 
                                                             core_config["final_flood_dates"], 
                                                             val_ratio, 
                                                             test_ratio)
    
    save_files(data_config_file, resolution, train_files,'training')
    save_files(data_config_file, resolution, val_files, 'validation')
    save_files(data_config_file, resolution, test_files, 'test')
    