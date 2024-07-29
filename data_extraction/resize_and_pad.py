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
    drive_files = generate_files_of_interest(drive, batch, "static/config.json", 'MSWEP_Past_3hr_folder_id')

    for j, ts in enumerate(batch):
        rainfall_file = pull_and_crop_rainfall_data(drive, 
                                                    drive_files[j], 
                                                    ts, 
                                                    "./data/temp/", 
                                                    bounding_box, 
                                                    "static/config.json", 
                                                    desired_resolution)
        destination = os.path.join(config[f"rainfall_path_{desired_resolution}_{desired_resolution}"], os.path.basename(rainfall_file))
        print(rainfall_file, destination)
        shutil.move(rainfall_file, destination)


if __name__ == "__main__":
    # Tweak water, soil moisture, topology
    desired_resolution = 256
    with open("static/config.json") as config_file:
        config = json.load(config_file)
    
    resize_and_pad_file_paths = [config['non_flood_file_path_2044_2573'],
                                 config['flood_file_path_2044_2573'],
                                 config['soil_moisture_flood_path_2044_2573'],
                                 config['soil_moisture_non_flood_path_2044_2573'],
                                 config['topology_path_2044_2573']]
    target_file_paths = [config['non_flood_file_path'],
                         config['flood_file_path'],
                         config['soil_moisture_flood_path'],
                         config['soil_moisture_non_flood_path'],
                         config['topology_path']]
    
    target_file_paths = [f"{config['non_flood_file_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
                         f"{config['flood_file_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
                         f"{config['soil_moisture_flood_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
                         f"{config['soil_moisture_non_flood_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
                         f"{config['topology_path']}_{str(desired_resolution)}_{str(desired_resolution)}"]
    
    for i in range(len(resize_and_pad_file_paths)):
        for f in os.listdir(resize_and_pad_file_paths[i]):
            raw_file_path = os.path.join(resize_and_pad_file_paths[i], f)
            # print("working on", raw_file_path)
            target_file_path = os.path.join(target_file_paths[i], f)
            resize_and_pad_with_PIL(raw_file_path, "static/config.json", desired_resolution, target_file_path)
    
    # Combine soil moisture and water images into one folder
    for i in range(len(target_file_paths[:-1])):
        # Water images
        if i < 2:
            combo_path = f"{config['water_image_path']}_{str(desired_resolution)}_{str(desired_resolution)}"
            os.makedirs(combo_path, exist_ok=True)
            for f in os.listdir(target_file_paths[i]):
                source = os.path.join(target_file_paths[i], f)
                destination = os.path.join(combo_path, f)
                shutil.copy(source, destination)
        # Soil Moisture images
        else:
            combo_path = f"{config['soil_moisture_combo_path']}_{str(desired_resolution)}_{str(desired_resolution)}"
            os.makedirs(combo_path, exist_ok=True)
            for f in os.listdir(target_file_paths[i]):
                source = os.path.join(target_file_paths[i], f)
                destination = os.path.join(combo_path, f)
                shutil.copy(source, destination)

    
    # # Tweak rainfall
    gauth = GoogleAuth()
    google_drive_credentials_path = config['google_drive_credentials']
    google_drive_oauth_path = config['google_drive_oauth']
    gauth.LoadClientConfigFile(google_drive_credentials_path)
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    bangladesh_shape = generate_country_outline('static/bangladesh-outline_68.geojson')
    bangladesh_bounding_box = box(*bangladesh_shape.bounds)


    rainfall_archive = config['rainfall_path_archive']

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
