import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from shapely.geometry import box
import pandas as pd
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_extraction.generic_helpers import *
from data_extraction.rainfall_helpers import *


def retry_function(func, args, kwargs, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            print(f'Attempt {attempt + 1} generated an exception: {exc}')
            if attempt + 1 < retries:
                print(f'Retrying in {delay} seconds...')
                time.sleep(delay)
            else:
                print('Maximum retries reached. Giving up.')
                raise


def generate_new_master(core_config_path: str, target_resolution: int):
    with open(core_config_path) as core_config_file:
        core_config = json.load(core_config_file)
    start_res_path = core_config[f'rainfall_reprojection_master_high_res']
    
    try: # Check if we already have this resolution available.
        resized_image_path = core_config[f'rainfall_reprojection_master_{desired_resolution}']
        return
    except KeyError: #If not, perform resizing
        original_image = Image.open(start_res_path)
        # Resize the image to new dimensions
        original_width, original_height = original_image.size

        target_resolution_is_height = True
        if original_height < original_width:
            target_resolution_is_height = False

        aspect_ratio = original_width / original_height
        if target_resolution_is_height:
            new_height = target_resolution
            new_width = int(aspect_ratio * new_height)
        else:
            new_width = target_resolution
            new_height = int(new_width / aspect_ratio)

        
        resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)

        # Save the resized image
        pattern = r'(\d+)_(\d+)\.tif$'
        replacement = f'{new_width}_{new_height}.tif'
        resized_image_path = re.sub(pattern, replacement, start_res_path)
        resized_image.save(resized_image_path)

        #Update config file
        core_config[f'rainfall_reprojection_master_{desired_resolution}'] = resized_image_path
        with open(core_config_path, 'w') as core_config_file:
            json.dump(core_config, core_config_file, indent=4)
    

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


    # Check if the directory exists, if not create it. data_config is already present i the namespace above this function level
    directory = f"{data_config['rainfall_path']}_{desired_resolution}_{desired_resolution}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for j, ts in enumerate(batch):
        rainfall_file = pull_and_crop_rainfall_data(drive, 
                                                    drive_files[j], 
                                                    ts, 
                                                    os.environ["PROJECT_FLOOD_DATA"], 
                                                    bounding_box, 
                                                    os.environ["PROJECT_FLOOD_CORE_PATHS"], 
                                                    desired_resolution)
        destination = os.path.join(f"{data_config['rainfall_path']}_{desired_resolution}_{desired_resolution}", os.path.basename(rainfall_file))
        # print(rainfall_file, destination)
        shutil.move(rainfall_file, destination)


if __name__ == "__main__":
    # Tweak water, soil moisture, topology
    desired_resolution = 128

    generate_new_master(os.environ["PROJECT_FLOOD_CORE_PATHS"], desired_resolution)

    with open(os.environ["PROJECT_FLOOD_DATA"]) as data_config_file:
        data_config = json.load(data_config_file)
    
    resize_and_pad_file_paths = [data_config['non_flood_file_path_2044_2573'],
                                 data_config['flood_file_path_2044_2573'],
                                 data_config['soil_moisture_flood_path_2044_2573'],
                                 data_config['soil_moisture_non_flood_path_2044_2573'],
                                 data_config['topology_path_2044_2573']]
    
    target_file_paths = [f"{data_config['non_flood_file_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
                         f"{data_config['flood_file_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
                         f"{data_config['soil_moisture_flood_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
                         f"{data_config['soil_moisture_non_flood_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
                         f"{data_config['topology_path']}_{str(desired_resolution)}_{str(desired_resolution)}"]
    
    # resize_and_pad_file_paths = [data_config['non_flood_file_path_2044_2573'],
    #                              data_config['flood_file_path_2044_2573'],
    #                              data_config['soil_moisture_flood_path_2044_2573'],
    #                              data_config['soil_moisture_non_flood_path_2044_2573'],
    #                              data_config['soil_moisture_watermask_flood_path_2044_2573'],
    #                              data_config['soil_moisture_watermask_non_flood_path_2044_2573']]

    
    # target_file_paths = [f"{data_config['non_flood_file_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
    #                      f"{data_config['flood_file_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
    #                      f"{data_config['soil_moisture_flood_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
    #                      f"{data_config['soil_moisture_non_flood_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
    #                      f"{data_config['soil_moisture_watermask_flood_path']}_{str(desired_resolution)}_{str(desired_resolution)}",
    #                      f"{data_config['soil_moisture_watermask_non_flood_path']}_{str(desired_resolution)}_{str(desired_resolution)}"]

    # Do the resizing and padding
    for i in range(len(resize_and_pad_file_paths)):
        for f in os.listdir(resize_and_pad_file_paths[i]):
            raw_file_path = os.path.join(resize_and_pad_file_paths[i], f)
            # print("working on", raw_file_path)
            target_file_path = os.path.join(target_file_paths[i], f)
            resize_and_pad_with_PIL(raw_file_path, os.environ["PROJECT_FLOOD_CORE_PATHS"], desired_resolution, target_file_path)
    
    # Combine soil moisture and water images into one folder
    for i in range(len(target_file_paths)):
        if 'Topology' in target_file_paths[i]:
            continue
        # Water images
        if 'FloodImages' in target_file_paths[i]:
        # if i < 2:
            combo_path = f"{data_config['water_image_path']}_{str(desired_resolution)}_{str(desired_resolution)}"
            os.makedirs(combo_path, exist_ok=True)
            for f in os.listdir(target_file_paths[i]):
                source = os.path.join(target_file_paths[i], f)
                destination = os.path.join(combo_path, f)
                shutil.copy(source, destination)
        # Soil Moisture images
        elif 'watermask' in target_file_paths[i]:
            combo_path = f"{data_config['soil_moisture_watermask_combo_path']}_{str(desired_resolution)}_{str(desired_resolution)}"
            os.makedirs(combo_path, exist_ok=True)
            for f in os.listdir(target_file_paths[i]):
                source = os.path.join(target_file_paths[i], f)
                destination = os.path.join(combo_path, f)
                shutil.copy(source, destination)
        else:
            combo_path = f"{data_config['soil_moisture_combo_path']}_{str(desired_resolution)}_{str(desired_resolution)}"
            os.makedirs(combo_path, exist_ok=True)
            for f in os.listdir(target_file_paths[i]):
                source = os.path.join(target_file_paths[i], f)
                destination = os.path.join(combo_path, f)
                shutil.copy(source, destination)

    
    # Tweak rainfall
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


    rainfall_archive = data_config['rainfall_integrity_path'] #list of files that we want to get from GDrive
    big_filedates = [pd.Timestamp(int(i[:4]), 1, 1) + pd.Timedelta(days=int(i[4:7]) - 1, hours=int(i[8:10])) 
                        for i in os.listdir(rainfall_archive)]
    
    target_directory = f"{data_config['rainfall_path']}_{desired_resolution}_{desired_resolution}"
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    small_filedates = [pd.Timestamp(int(i[:4]), 1, 1) + pd.Timedelta(days=int(i[4:7]) - 1, hours=int(i[8:10])) 
                        for i in os.listdir(target_directory)]
    
    filedates = [i for i in big_filedates if i not in small_filedates]
    print(len(filedates))
    batch_size = 10

    # filedates = [i for i in os.listdir("/home/vkhandekar/project_flood/data/BangladeshRainfall_256_256") if i not in os.listdir("/home/vkhandekar/project_flood/data/BangladeshRainfall_128_128")]
    # filedates = [pd.Timestamp(int(i[:4]), 1, 1) + pd.Timedelta(days=int(i[4:7]) - 1, hours=int(i[8:10])) for i in filedates]
    max_workers = 6

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_rainfall_batch, filedates[i:i+batch_size], drive, bangladesh_bounding_box) for i in range(0, len(filedates), batch_size)]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                # print(f'Result: {result}')
            except Exception as exc:
                print(f'Generated an exception: {exc}')

    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [
    #         executor.submit(retry_function, process_rainfall_batch, (filedates[i:i + batch_size], drive, bangladesh_bounding_box), {}, retries=3, delay=5) 
    #             for i in range(0, len(filedates), batch_size)
    #         ]

    #     for future in as_completed(futures):
    #         try:
    #             result = future.result()
    #             # print(f'Result: {result}')
    #         except Exception as exc:
    #             print(f'Final failure, generated an exception: {exc}')
