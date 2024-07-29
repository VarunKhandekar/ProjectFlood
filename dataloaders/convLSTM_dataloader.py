import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from data_extraction.generic_helpers import *
from dataloaders.custom_image_transforms import *

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


def normalise_rainfall(image, min, max):
    return (image - min)/(max - min)


def standardise_locally(image, thres_roi = 1.0):
    val_l = np.percentile(image, thres_roi)
    roi = (image >= val_l)
    mu, sigma = np.mean(image[roi]), np.std(image[roi])
    eps = 1e-6
    image2 = (image - mu) / (sigma + eps)
    return image2