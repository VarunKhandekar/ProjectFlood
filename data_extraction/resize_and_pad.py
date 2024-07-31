import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from shapely.geometry import box
import pandas as pd
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_extraction.generic_helpers import *
from data_extraction.rainfall_helpers import *

def process_rainfall_batch(batch: list[pd.Timestamp], drive: GoogleDrive, bounding_box: box):
    """
    Process a batch of rainfall data files by pulling, cropping, and saving them to the desired location.

    Args:
        batch (list[pd.Timestamp]): A list of pandas timestamps for which to process the rainfall data.
        drive (GoogleDrive): Authenticated GoogleDrive object.
        bounding_box (box): A shapely box object defining the area to crop the data to.

    This function performs the following steps:
        1. Generates a list of files of interest from Google Drive based on the timestamps in the batch.
        2. For each timestamp in the batch:
            a. Pulls and crops the corresponding rainfall data file from Google Drive.
            b. Moves the processed file to the specified destination directory.
    
    Notes:
        - The configuration file (`static/config.json`) should contain the necessary paths and folder IDs.
        - The desired resolution for the processed files should be specified in the `desired_resolution` variable.
    """
    drive_files = generate_files_of_interest(drive, batch, os.environ["PROJECT_FLOOD_CORE_PATHS"], 'MSWEP_Past_3hr_folder_id')

    for j, ts in enumerate(batch):
        rainfall_file = pull_and_crop_rainfall_data(drive, 
                                                    drive_files[j], 
                                                    ts, 
                                                    os.environ["PROJECT_FLOOD_DATA"], 
                                                    bounding_box, 
                                                    os.environ["PROJECT_FLOOD_CORE_PATHS"], 
                                                    desired_resolution)
        destination = os.path.join(f"{data_config['rainfall_path']}_{desired_resolution}_{desired_resolution}", os.path.basename(rainfall_file))
        print(rainfall_file, destination)
        shutil.move(rainfall_file, destination)


if __name__ == "__main__":
    # Tweak water, soil moisture, topology
    desired_resolution = 256
    with open(os.environ["PROJECT_FLOOD_DATA"]) as data_config_file:
        data_config = json.load(data_config_file)
    
    resize_and_pad_file_paths = [data_config['non_flood_file_path_2044_2573'],
                                 data_config['flood_file_path_2044_2573'],
                                 data_config['soil_moisture_flood_path_2044_2573'],
                                 data_config['soil_moisture_non_flood_path_2044_2573'],
                                 data_config['topology_path_2044_2573']]
    target_file_paths = [data_config['non_flood_file_path'],
                         data_config['flood_file_path'],
                         data_config['soil_moisture_flood_path'],
                         data_config['soil_moisture_non_flood_path'],
                         data_config['topology_path']]
    
    target_file_paths = [f"{data_config['non_flood_file_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
                         f"{data_config['flood_file_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
                         f"{data_config['soil_moisture_flood_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
                         f"{data_config['soil_moisture_non_flood_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
                         f"{data_config['topology_path']}_{str(desired_resolution)}_{str(desired_resolution)}"]
    
    for i in range(len(resize_and_pad_file_paths)):
        for f in os.listdir(resize_and_pad_file_paths[i]):
            raw_file_path = os.path.join(resize_and_pad_file_paths[i], f)
            # print("working on", raw_file_path)
            target_file_path = os.path.join(target_file_paths[i], f)
            resize_and_pad_with_PIL(raw_file_path, os.environ["PROJECT_FLOOD_CORE_PATHS"], desired_resolution, target_file_path)
    
    # Combine soil moisture and water images into one folder
    for i in range(len(target_file_paths[:-1])):
        # Water images
        if i < 2:
            combo_path = f"{data_config['water_image_path']}_{str(desired_resolution)}_{str(desired_resolution)}"
            os.makedirs(combo_path, exist_ok=True)
            for f in os.listdir(target_file_paths[i]):
                source = os.path.join(target_file_paths[i], f)
                destination = os.path.join(combo_path, f)
                shutil.copy(source, destination)
        # Soil Moisture images
        else:
            combo_path = f"{data_config['soil_moisture_combo_path']}_{str(desired_resolution)}_{str(desired_resolution)}"
            os.makedirs(combo_path, exist_ok=True)
            for f in os.listdir(target_file_paths[i]):
                source = os.path.join(target_file_paths[i], f)
                destination = os.path.join(combo_path, f)
                shutil.copy(source, destination)

    
    # # Tweak rainfall
    gauth = GoogleAuth()
    with open(os.environ["PROJECT_FLOOD_CORE_PATHS"]) as core_config_file:
        core_config = json.load(core_config_file)
    google_drive_credentials_path = core_config['google_drive_credentials']
    google_drive_oauth_path = core_config['google_drive_oauth']
    gauth.LoadClientConfigFile(google_drive_credentials_path)
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    bangladesh_shape = generate_country_outline(core_config["bangladesh_shape_outline"])
    bangladesh_bounding_box = box(*bangladesh_shape.bounds)


    rainfall_archive = data_config['rainfall_path_archive']

    filedates = [pd.Timestamp(int(i[:4]), 1, 1) + pd.Timedelta(days=int(i[4:7]) - 1, hours=int(i[8:10])) 
                 for i in os.listdir(rainfall_archive)]
    print(len(filedates))
    batch_size = 10

    max_workers = 6
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_rainfall_batch, filedates[i:i+batch_size], drive, bangladesh_bounding_box) for i in range(0, len(filedates), batch_size)]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                # print(f'Result: {result}')
            except Exception as exc:
                print(f'Generated an exception: {exc}')
