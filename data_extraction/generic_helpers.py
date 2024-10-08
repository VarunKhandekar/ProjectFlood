import json
import pandas as pd
import numpy as np
import os
import re
import rasterio
from datetime import timedelta
from shapely.geometry import Polygon
from PIL import Image, ImageOps


def generate_country_outline(shapefile_path: str) -> Polygon:
    """
    Generate the outline of a country from a GeoJSON shapefile.

    Args:
        shapefile_path (str): Path to the GeoJSON shapefile.

    Returns:
        Polygon: A shapely Polygon object representing the country's outline.
    """
    with open(shapefile_path, 'r') as file:
        geojson_data = json.load(file)

    # Extract polygon coordinates
    polygon_coordinates = []

    for feature in geojson_data['features']:
        if feature['geometry']['type'] == 'Polygon':
            polygon_coordinates.append(feature['geometry']['coordinates'])
        elif feature['geometry']['type'] == 'MultiPolygon':
            for polygon in feature['geometry']['coordinates']:
                polygon_coordinates.append(polygon)
    
    return Polygon(polygon_coordinates[0][0])


def generate_square_coordinates_from_polygon(polygon: Polygon) -> tuple:
    """
    Generate a square bounding box around a given polygon by expanding the smaller dimension to match the larger one.

    Args:
        polygon (Polygon): A shapely Polygon object representing the input polygon.

    Returns:
        tuple: A tuple of four float values representing the coordinates of the square bounding box:
            (new_min_lon, new_min_lat, new_max_lon, new_max_lat).

    """
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    width = max_lon - min_lon
    height = max_lat - min_lat
    square_size = max(width, height)

    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    new_min_lon = center_lon - square_size / 2
    new_min_lat = center_lat - square_size / 2
    new_max_lon = center_lon + square_size / 2
    new_max_lat = center_lat + square_size / 2
    return new_min_lon, new_min_lat, new_max_lon, new_max_lat


def generate_timestamps(date: pd.Timestamp, days_before: int, days_after: int, freq: str) -> list:
    """
    Generate a list of timestamps around a given date.

    Args:
        date (pd.Timestamp): The central date.
        days_before (int): Number of days before the central date to start generating timestamps.
        days_after (int): Number of days after the central date to stop generating timestamps.
        freq (str): Frequency string for the timestamps (e.g., 'H' for hourly, 'D' for daily).

    Returns:
        list: A list of timestamps.
    """
    start_time = date - pd.Timedelta(days=days_before)
    end_time = date + pd.Timedelta(days=days_after)

    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
    # don't include final forecast as it takes us into day t+days_after+1
    return timestamps.tolist()[:-1]


def generate_random_non_flood_dates(core_config_path: str, num_dates: int, safety_window: int, data_config_path: str) -> list:
    """
    Generate a list of random dates that are not during flood periods.

    Args:
        file_path (str): Path to the data file (CSV or Excel) containing flood periods.
        num_dates (int): Number of random non-flood dates to generate.
        safety_window (int): Safety buffer around flood periods in days.
        config_file (str): Path to the configuration file containing existing non-flood dates.

    Returns:
        list: A list of random non-flood dates.
    """
    with open(core_config_path) as core_config_file:
        core_config = json.load(core_config_file)
    file_path = core_config["flood_events"]

    # Read in data file
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError("Please provide a CSV or Excel file.")

    # Data file integrity checks
    required_columns = ['dfo_began_uk', 'dfo_ended_uk']
    assert all(column in df.columns for column in required_columns), f"Missing columns: {[column for column in required_columns if column not in df.columns]}"

    df['dfo_began_uk'] = pd.to_datetime(df['dfo_began_uk'])
    df['dfo_ended_uk'] = pd.to_datetime(df['dfo_ended_uk'])

    # Define start and end dates for random date generation
    start_date = df['dfo_began_uk'].min()
    end_date = df['dfo_ended_uk'].max()

    # Define periods to avoid looking at
    exclusion_periods = list(zip(df['dfo_began_uk'], df['dfo_ended_uk']))

    # Get current dates we have already downloaded
    with open(data_config_path) as data_config_file:
        data_config = json.load(data_config_file)
    existing_image_file_path = f"{data_config['non_flood_file_path']}_256_256"
    current_dates = [pd.to_datetime(re.search(r'\d{8}', i).group(0), format=r'%Y%m%d') for i in os.listdir(existing_image_file_path)]

    random_dates = []
    while len(random_dates) <= num_dates:
        random_date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))
        # Make sure date is not during any of the periods to be avoided i.e. when there was a flood
        # Include a safety buffer around the start and end dates
        if all(not(period_start - timedelta(days=safety_window) <= random_date <= period_end + timedelta(days=safety_window)) 
               for period_start, period_end in exclusion_periods):
            
            # Check we don't already have this date
            if random_date not in current_dates:
                random_dates.append(random_date)
    
    return random_dates


def remove_metadata(image_path: str) -> None:
    """
    Remove metadata from an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        None
    """
    image = Image.open(image_path)
    data = list(image.getdata())
    image_without_metadata = Image.new(image.mode, image.size)
    image_without_metadata.putdata(data)
    image_without_metadata.save(image_path)


def pad_to_square(image: Image, desired_resolution: int) -> Image:
    """
    Pad an image to make it square with specified dimensions.

    Args:
        image (Image): The input image to be padded.
        desired_resolution (int): The desired width and height of the padded image.

    Returns:
        Image: The padded image with the specified dimensions.
    """
    pad_width = desired_resolution - image.width
    pad_height = desired_resolution - image.height
    padded_image = ImageOps.expand(image, (0, 0, pad_width, pad_height), fill='black')
    return padded_image


def resize_and_pad_with_PIL(file_path: str, core_config_path: str, desired_resolution: int, target_file_path: str) -> None:
    """
    Resize and pad an image using the Python Imaging Library (PIL), and save the processed image to a specified target path.

    Args:
        file_path (str): Path to the input image file to be resized and padded.
        core_config_path (str): Path to the core configuration JSON file containing the paths for reference images.
        desired_resolution (int): Desired resolution for padding the image.
        target_file_path (str): Path to save the resized and padded image.

    Returns:
        None

    """
    with open(core_config_path) as core_config_file:
        core_config = json.load(core_config_file)
    target_image_file_path = core_config[f'rainfall_reprojection_master_{desired_resolution}']
    target_image = Image.open(target_image_file_path)
    new_width, new_height = target_image.width, target_image.height

    directory = os.path.dirname(target_file_path)

    # Check if the directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        with Image.open(file_path) as img:
            resized_img = img.resize((new_width, new_height))
            resized_and_padded_img = pad_to_square(resized_img, desired_resolution)
            resized_and_padded_img.save(target_file_path, compression='tiff_deflate')
    except:
        img = rasterio.open(file_path)
        img = img.read(1)
        img = Image.fromarray(img)
        resized_img = img.resize((new_width, new_height))
        resized_and_padded_img = pad_to_square(resized_img, desired_resolution)
        resized_and_padded_img.save(target_file_path, compression='tiff_deflate')
