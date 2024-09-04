import os
import json
import subprocess
from data_extraction.generic_helpers import *
from data_extraction.rainfall_helpers import *

def check_relevant_files_exist(core_config_path: str, data_config_path: str, days_before: int, days_after: int, freq: str) -> tuple:
    """
    Check if all relevant rainfall and soil moisture files exist for flood and non-flood images, and return missing files and rerun dates.

    Args:
        core_config_path (str): Path to the core configuration JSON file containing cleanup script information.
        data_config_path (str): Path to the data configuration JSON file containing file paths for flood, non-flood, rainfall, and soil moisture images.
        days_before (int): Number of days before the image date for which rainfall data needs to be checked.
        days_after (int): Number of days after the image date for which rainfall data needs to be checked.
        freq (str): Frequency for generating the list of timestamps for rainfall data.

    Returns:
        tuple: A tuple containing three elements:
            - missing_rainfall_images (list): List of rainfall image file names that are missing.
            - missing_soil_moisture_images (list): List of soil moisture image file names that are missing.
            - rerun_dates (list): List of dates that need to be rerun based on missing images.

    """
    # Call script to ensure folders are cleaned up
    with open(core_config_path) as core_config_file:
        core_config = json.load(core_config_file)
    result = subprocess.run([core_config['cleanup']], capture_output=True, text=True)

    with open(data_config_path) as data_config_file:
        data_config = json.load(data_config_file)

    missing_rainfall_images, missing_soil_moisture_images, rerun_dates = [], [], []

    # Go through flood and non-flood images.
    water_images = []
    water_images.extend(os.listdir(data_config["non_flood_file_path_2044_2573"]))
    water_images.extend(os.listdir(data_config["flood_file_path_2044_2573"]))

    soil_moisture_images = []
    soil_moisture_images.extend(os.listdir(data_config["soil_moisture_flood_path_2044_2573"]))
    soil_moisture_images.extend(os.listdir(data_config["soil_moisture_non_flood_path_2044_2573"]))

    
    for i in water_images:
        # For each, extract list of relevant rainfall images. Check they exist. If one doesn't, add it to missing rainfall
        date = pd.to_datetime(i[-12:-4], format=r'%Y%m%d')
        timestamps = generate_timestamps(date, days_before, days_after, freq)
        rainfall_file_names = [convert_date_to_MSWEP_file_name(t) for t in timestamps]
        rainfall_file_names = [rf[:-2] + "tif" for rf in rainfall_file_names]

        for rf in rainfall_file_names:
            if rf not in os.listdir(data_config["rainfall_integrity_path"]):
                # print(rf, " is missing!")
                missing_rainfall_images.append(rf)
                rerun_dates.append(date)

        # For each, extract list of relevant soil moisture. Check it exists. If one doesn't, add it to missing soil moisture
        soil_moisture_date = date - pd.Timedelta(days=1)
        soil_moisture_name = f"BangladeshSoilMoisture{soil_moisture_date.strftime(r'%Y%m%d')}.tif"
        if soil_moisture_name not in soil_moisture_images:
            print(soil_moisture_name, " is missing!")
            missing_soil_moisture_images.append(soil_moisture_name)
            rerun_dates.append(date)

    rerun_dates = list(set(rerun_dates))
    for i in rerun_dates:
        print(i, "needs to be rerun")

    return missing_rainfall_images, missing_soil_moisture_images, rerun_dates


if __name__=="__main__":
    # print(os.getcwd())
    # print(os.listdir("data/BangladeshRainfall"))
    missing_rainfall_images, missing_soil_moisture_images, rerun_dates = check_relevant_files_exist(os.environ["PROJECT_FLOOD_CORE_PATHS"], os.environ["PROJECT_FLOOD_DATA"], 7, 1, "3h")
    print(len(missing_rainfall_images))
    print(len(rerun_dates))
    print(rerun_dates)

    rerun_dates_series = pd.Series(rerun_dates)
    rerun_dates_series.to_csv("static/to_rerun.csv", index=False, header=False)