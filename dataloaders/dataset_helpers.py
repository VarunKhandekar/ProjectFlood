import imageio.v2 as imageio
import torch
import pandas as pd
import os
from data_extraction.generic_helpers import *

def get_log_rainfall_stats_training(training_path: str, rainfall_dir: str, preceding_rainfall_days: int, forecast_rainfall_days: int = 1):
    minimum, maximum = 0.0, 0.0
    
    for im in os.listdir(training_path):
        date_str = im[15:-4]
        date = pd.to_datetime(date_str, format=r"%Y%m%d")
        
        rainfall_dates = generate_timestamps(date, preceding_rainfall_days, forecast_rainfall_days, "3h")
        for rd in rainfall_dates:
            rain_image_name = os.path.join(rainfall_dir, rd.strftime(r"%Y%j.%H")+".tif")
            rain_image = imageio.imread(rain_image_name)
            rain_image = rain_image.astype(np.float32) # De quantize
            rain_image /= 1000.0
            rain_image = np.log(rain_image + 1) #Take log
            maximum = np.maximum(maximum, np.max(rain_image))
            minimum = np.minimum(minimum, np.min(rain_image))

    return minimum, maximum

# def get_rainfall_stats_training(training_path: str, rainfall_dir: str, preceding_rainfall_days: int, forecast_rainfall_days: int = 1):
#     minimum, maximum = 0.0, 0.0
    
#     for im in os.listdir(training_path):
#         date_str = im[15:-4]
#         date = pd.to_datetime(date_str, format=r"%Y%m%d")
        
#         rainfall_dates = generate_timestamps(date, preceding_rainfall_days, forecast_rainfall_days, "3h")
#         for rd in rainfall_dates:
#             rain_image_name = os.path.join(rainfall_dir, rd.strftime(r"%Y%j.%H")+".tif")
#             rain_image = imageio.imread(rain_image_name)
#             rain_image = rain_image.astype(np.float32) # De quantize
#             rain_image /= 1000.0
#             # rain_image = np.log(rain_image + 1) #Take log
#             maximum = np.maximum(maximum, np.max(rain_image))
#             minimum = np.minimum(minimum, np.min(rain_image))

#     return minimum, maximum


def normalise_rainfall(image, min, max):
    return (image - min)/(max - min)


def standardise_locally(image, thres_roi = 1.0):
    val_l = np.percentile(image, thres_roi)
    roi = (image >= val_l)
    mu, sigma = np.mean(image[roi]), np.std(image[roi])
    eps = 1e-6
    image2 = (image - mu) / (sigma + eps)
    return image2

def generate_label_images(label_name, soil_moisture_dir, topology_dir, rainfall_dir, rainfall_min, rainfall_max, preceding_rainfall_days, forecast_rainfall_days = 1):

    # Get 'conditioning' images and collate into a dictionary
    # Extract date from file path
    images_dict = {}
    date_str = label_name[15:-4]
    date = pd.to_datetime(date_str, format=r"%Y%m%d")

    #Topology - standardisation, float32 at end
    topology_name = os.path.join(topology_dir, "BangladeshTopology.tif")
    # images_dict['topology'] = imageio.imread(topology_name).toTensor()
    topology_image = imageio.imread(topology_name)
    topology_image = topology_image.astype(np.float32)
    # topology_image = standardise_locally(topology_image)
    topology_image = (topology_image - np.min(topology_image)) / (np.max(topology_image) - np.min(topology_image)) # Scaling
    images_dict['topology'] = [topology_image]

    #Soil Moisture - pseudo min-max scaling, float32 at the end
    soil_moisture_date = date - pd.Timedelta(days=1)
    soil_moisture_name = os.path.join(soil_moisture_dir, 
                                        "BangladeshSoilMoisture" + soil_moisture_date.strftime(r"%Y%m%d") + ".tif")
    soil_moisture_image = imageio.imread(soil_moisture_name)
    soil_moisture_image = soil_moisture_image.astype(np.float32) #coerce to float32 just in case
    soil_moisture_image = np.clip(soil_moisture_image, 0, 1) # pseudo-mix-max-scaling!
    images_dict['soil_moisture'] = [soil_moisture_image]

    #Rainfall - log min-max scaling, float32 at the end
    rainfall_dates = generate_timestamps(date, preceding_rainfall_days, forecast_rainfall_days, "3h")
    preceding = []
    forecast = []
    for rd in rainfall_dates:
        rain_image_name = os.path.join(rainfall_dir, rd.strftime(r"%Y%j.%H")+".tif")
        rain_image = imageio.imread(rain_image_name)
        rain_image = rain_image.astype(np.float32) # De quantize
        rain_image /= 1000.0
        rain_image = np.log(rain_image + 1) #Take log
        rain_image = normalise_rainfall(rain_image, rainfall_min, rainfall_max)
        rain_image = np.clip(rain_image, 0, 1) # FOR HANDLING CASES WHERE TEST/VALIDATION DATA EXCEEDS THE RANGE
        if rd < date:
            preceding.append(rain_image)
        else:
            forecast.append(rain_image)
    
    images_dict['preceding'] = preceding
    images_dict['forecast'] = forecast
    
    return images_dict

def prepare_tensors(tensor_list):
    if len(tensor_list) > 1:
        # Stack if there's more than one tensor
        return torch.stack(tensor_list, dim=0)
    else:
        # Add a dimension to make it [1, X, Y] if there's only one tensor
        return tensor_list[0].unsqueeze(0)