import json
import pandas as pd
from pydrive.drive import GoogleDrive
import os
import xarray as xr
from shapely.geometry import Polygon, mapping
# from osgeo import gdal
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import numpy as np
from PIL import Image
from generic_helpers import *

def convert_date_to_MSWEP_file_name(timestamp: pd.Timestamp) -> str:
    """
    Convert a timestamp to the corresponding MSWEP file name.

    Args:
        timestamp (pd.Timestamp): The timestamp to convert.

    Returns:
        str: The MSWEP file name corresponding to the given timestamp.
    """
    return f"{timestamp.year}{timestamp.dayofyear:03}.{timestamp.hour:02}.nc"

def generate_files_of_interest(drive: GoogleDrive, 
                               timestamps: list, 
                               core_config_path: str, 
                               folder: str) -> list:
    """
    Generate a list of files of interest from Google Drive based on given timestamps.

    Args:
        drive (GoogleDrive): Authenticated GoogleDrive object.
        timestamps (list): List of timestamps to generate file names for.
        config (str): Path to the configuration file containing Google Drive folder IDs.
        folder (str): The target folder identifier in the configuration file.

    Returns:
        list: A list of files from Google Drive that match the generated file names.
    """
    with open(core_config_path) as core_config_file:
        core_config = json.load(core_config_file)
    folder_id = core_config[folder]

    file_names = [convert_date_to_MSWEP_file_name(timestamp) for timestamp in timestamps]

    files = []
    for file_name in file_names:
        query = f"title='{file_name}' and '{folder_id}' in parents"
        file_list = drive.ListFile({'q': query}).GetList()
        files.extend(file_list)
    return files

def reproject_and_upsample_rasterio(input_file: str, 
                                    output_file: str, 
                                    core_config_path: str) -> str:
    """
    Reproject and upsample a raster file using Rasterio.

    Args:
        input_file (str): Path to the input raster file.
        output_file (str): Path to the output raster file.
        config_file (str): Path to the configuration file containing master file information.

    Returns:
        str: Path to the output file.

    Note:
        This function uses bilinear resampling and compresses the output using LZW compression.
    """
    with rasterio.open(input_file) as src:
        src_transform = src.transform
        data = src.read()
        src_crs = src.crs
        with open(core_config_path) as core_config_file:
            core_config = json.load(core_config_file)
        master_file = core_config['rainfall_reprojection_master']
        # Open target file to get target dimensions and transform
        with rasterio.open(master_file) as target:
            target_transform = target.transform
            target_width = target.width
            target_height = target.height
            target_crs = target.crs
            
            # Update metadata for the new file, including LZW compression
            kwargs = src.meta
            kwargs.update({
                'height': target_height,
                'width': target_width,
                'transform': target_transform,
                'crs': target_crs,
                'compress': 'lzw'
            })
        
        # Write to output
        with rasterio.open(output_file, 'w', **kwargs) as dst:
            for i, band in enumerate(data, 1): # Go through bands
                dest = np.zeros((target_height, target_width), dtype=np.float32)
                
                reproject(
                    source=band,
                    destination=dest,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )

                dst.write(dest, indexes=i)


    return output_file

# def reproject_and_upsample_PIL(input_file: str, 
#                                output_file: str, 
#                                core_config_path: str,
#                                desired_resolution: int) -> str:
#     """
#     Reproject, upsample, and pad an image file using PIL.

#     Args:
#         input_file (str): Path to the input image file.
#         output_file (str): Path to the output image file.
#         config_file (str): Path to the configuration file containing master file information.
#         desired_resolution (int): The desired width and height for the output image.

#     Returns:
#         str: Path to the output file.

#     This function performs the following steps:
#         1. Opens the input image file.
#         2. Retrieves the master file information from the configuration file.
#         3. Upsamples the input image to match the dimensions of the master file.
#         4. Pads the upsampled image to the desired resolution.
#         5. Saves the padded image to the specified output file with TIFF deflate compression.
#     """
#     lowres_image = Image.open(input_file)
#     with open(core_config_path) as core_config_file:
#         core_config = json.load(core_config_file)
#     target_image = Image.open(core_config['rainfall_reprojection_master_low_res'])

#     upsampled_image = lowres_image.resize((target_image.width, target_image.height))
#     upsampled_image = pad_to_square(upsampled_image, desired_resolution)
#     upsampled_image.save(output_file, compression='tiff_deflate')

#     return output_file

def get_transform_from_xarray(data_array):
    # Assuming the coordinates are named 'lon' and 'lat' and are regularly spaced
    lon_res = data_array.lon[1] - data_array.lon[0]  # Longitude resolution
    lat_res = data_array.lat[1] - data_array.lat[0]  # Latitude resolution
    transform = from_origin(data_array.lon.min(), data_array.lat.max(), lon_res, abs(lat_res))
    return transform


def pull_and_crop_rainfall_data(drive: GoogleDrive, 
                                original_file, 
                                timestamp: pd.Timestamp, 
                                data_config_path: str, 
                                shape: Polygon, 
                                core_config_path: str,
                                desired_resolution: int) -> str:
    """
    Pull, crop, and process rainfall data from Google Drive.

    Args:
        drive (GoogleDrive): Authenticated GoogleDrive object.
        original_file (dict): Metadata of the original file to be downloaded from Google Drive.
        timestamp (pd.Timestamp): The timestamp corresponding to the file.
        temp_output_path (str): Path to the temporary output directory.
        shape (Polygon): A shapely Polygon object defining the area to crop the data to.
        core_config (str): Path to the configuration file.
        desired_resolution (int): The desired width and height for the output image.

    Returns:
        str: Path to the processed output file.

    This function performs the following steps:
        1. Downloads the original NetCDF file from Google Drive.
        2. Clips the data to the specified geographic shape.
        3. Converts the data to a TIFF file.
        4. Reprojects, upsamples, and compresses the TIFF file to the desired resolution.
    """
    with open(data_config_path) as data_config_file:
        data_config = json.load(data_config_file)
    temp_output_path = data_config["temp_rainfall_path"]

    new_file = drive.CreateFile({'id': original_file['id']})
    new_path = os.path.join(temp_output_path, f"{timestamp.year}{timestamp.dayofyear:03}.{timestamp.hour:02}.nc")
    new_file.GetContentFile(new_path) # save down the file from google drive

    lowres_tif_file = os.path.join(temp_output_path, f"{timestamp.year}{timestamp.dayofyear:03}.{timestamp.hour:02}_lowres.tif")

    # Clip the data and save down into lowres_file_path
    geojson = [mapping(shape)] # Use GeoJSON of the shape
    with xr.open_dataset(new_path, engine="netcdf4") as data:
        data = data['precipitation']
        data.rio.set_spatial_dims('lon', 'lat', inplace=True)
        data.rio.write_crs("EPSG:4326", inplace=True)
        
        data_clipped = data.rio.clip(geojson, all_touched=True)
        data_clipped = ((data_clipped*1000)//1).astype(np.uint16) # QUANTIZATION
    data_clipped = np.squeeze(data_clipped) # get correct dimensions

    metadata = {
        'driver': 'GTiff',
        'height': data_clipped.shape[0],
        'width': data_clipped.shape[1],
        'count': 1,
        'dtype': data_clipped.dtype,  # Using the numpy dtype directly
        'crs': 'EPSG:4326', 
        'transform': get_transform_from_xarray(data_clipped)
    }
    with rasterio.open(lowres_tif_file, 'w', **metadata) as dst:
        dst.write(data_clipped.values, 1) 
    os.remove(new_path) #Remove original image from google drive
    
    # Do reprojection, upsampling and compression before saving down
    output_tif_file = os.path.join(temp_output_path, f"{timestamp.year}{timestamp.dayofyear:03}.{timestamp.hour:02}.tif")
    # reproject_and_upsample_rasterio(lowres_tif_file, output_tif_file, config_file) #2044x2573
    resize_and_pad_with_PIL(lowres_tif_file, core_config_path, desired_resolution, output_tif_file) #256x256 #Saves the resampled image
    os.remove(lowres_tif_file)

    return output_tif_file