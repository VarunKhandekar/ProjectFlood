import os
import json
import subprocess
from data_extraction.generic_helpers import *
from data_extraction.rainfall_helpers import *

def check_relevant_files_exist(config_file_path: str, days_before: int, days_after: int, freq: str):
    # Call script to ensure folders are cleaned up
    with open(config_file_path) as config_file:
        config = json.load(config_file)
    result = subprocess.run([config['cleanup']], capture_output=True, text=True)

    missing_rainfall_images, missing_soil_moisture_images, rerun_dates = [], [], []

    # Go through flood and non-flood images.
    water_images = []
    water_images.extend(os.listdir(config["non_flood_file_path"]))
    water_images.extend(os.listdir(config["flood_file_path"]))

    soil_moisture_images = []
    soil_moisture_images.extend(os.listdir(config["soil_moisture_flood_path"]))
    soil_moisture_images.extend(os.listdir(config["soil_moisture_non_flood_path"]))

    
    for i in water_images:
        # For each, extract list of relevant rainfall images. Check they exist. If one doesn't, add it to missing rainfall
        date = pd.to_datetime(i[-12:-4], format=r'%Y%m%d')
        timestamps = generate_timestamps(date, days_before, days_after, freq)
        rainfall_file_names = [convert_date_to_MSWEP_file_name(t) for t in timestamps]
        rainfall_file_names = [rf[:-2] + "tif" for rf in rainfall_file_names]

        for rf in rainfall_file_names:
            if rf not in os.listdir(config["rainfall_path"]):
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
    missing_rainfall_images, missing_soil_moisture_images, rerun_dates = check_relevant_files_exist("static/config.json", 7, 1, "3h")
    print(len(missing_rainfall_images))
    print(len(rerun_dates))
    print(rerun_dates)

    rerun_dates_series = pd.Series(rerun_dates)
    rerun_dates_series.to_csv("static/to_rerun.csv", index=False, header=False)