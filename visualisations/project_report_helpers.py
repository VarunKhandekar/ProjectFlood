import os
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import ee
from data_extraction.generic_helpers import *


def plot_bangladesh_flood_distribution(core_config_file_path: str, data_config_file_path: str) -> None:
    """
    Plot the distribution of flood events in Bangladesh by month and save the plot as an image file.

    Args:
        core_config_file_path (str): Path to the core configuration JSON file containing the flood events file path.
        data_config_file_path (str): Path to the data configuration JSON file containing the path for saving the plot.

    Returns:
        None: The function generates and saves a bar plot showing the total count of flood events in Bangladesh by month.

    Notes:
        The function reads the flood events data from an Excel file, calculates the total number of flood events per month,
        and saves the bar chart as a PNG image in the directory specified in the data configuration file.

    """
    with open(core_config_file_path) as core_config_file:
        core_config = json.load(core_config_file)
    
    with open(data_config_file_path) as data_config_file:
        data_config = json.load(data_config_file)

    floods_file_path = core_config["flood_events"]
    floods = pd.read_excel(floods_file_path)

    floods['month_name'] = floods['dfo_began_uk'].dt.strftime('%B')

    # Count the number of events per month name, summing across all years
    monthly_event_counts_across_years = floods['month_name'].value_counts().reindex(
        pd.date_range('2024-01-01', periods=12, freq='M').strftime('%B'))

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    monthly_event_counts_across_years.plot(kind='bar')
    plt.title('Total Count of Flood Events in Bangladesh by Month')
    plt.xlabel('Month')
    plt.ylabel('Total Count of Flood Events')
    plt.xticks(rotation=45)
    plt.grid(False)  # Remove gridlines
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)  # Remove x-tick marks

    output_file_name = os.path.join(data_config['project_report_visuals'], 'bangladesh_flood_distribution.png')
    plt.savefig(output_file_name, bbox_inches='tight')
    plt.show()


def plot_bangladesh_monthly_soil_moisture(data_config_file_path: str, shape: ee.geometry.Geometry) -> ee.ImageCollection:
    """
    Plot the average monthly soil moisture in Bangladesh from 2000 to 2018 (GFD dataset date range) and save the plot as an image file. 
    
    Soil moisture data is retrieved from the 'ECMWF/ERA5_LAND/MONTHLY_AGGR' dataset

    Args:
        data_config_file_path (str): Path to the data configuration JSON file containing the path for saving the plot.
        shape (ee.geometry.Geometry): The geographical region for which the soil moisture data will be analyzed.

    Returns:
        ee.ImageCollection: The image collection containing the monthly average soil moisture values.

    """
    with open(data_config_file_path) as data_config_file:
        data_config = json.load(data_config_file)
    # Define the date range based on GFD
    start_date = '2000-02-17'
    end_date = '2018-12-10'

    monthly_dataset = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').filterDate(start_date, end_date)
    soil_moisture_monthly = monthly_dataset.select('volumetric_soil_water_layer_1')

    def extract_month(image):
        date = ee.Date(image.get('system:time_start'))
        month = date.get('month')
        return image.set('month', month)

    soil_moisture_by_month = soil_moisture_monthly.map(extract_month)

    def calculate_monthly_mean(month):
        filtered = soil_moisture_by_month.filter(ee.Filter.eq('month', month))
        return filtered.mean().set('month', month)

    monthly_averages = ee.ImageCollection([calculate_monthly_mean(month) for month in range(1, 13)])
    clipped_monthly_averages = monthly_averages.map(lambda image: image.clip(shape))

    def reduce_and_mean(image):
        mean_value = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=shape, scale=1000)
        return image.set('mean_value', mean_value.get('volumetric_soil_water_layer_1'))

    final_means = clipped_monthly_averages.map(reduce_and_mean)
    data = final_means.getInfo()

    months_full = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    moisture_values = [entry['properties']['mean_value'] for entry in data['features']]

    plt.figure(figsize=(10, 6))
    plt.bar(months_full, moisture_values)
    plt.title('Average Monthly Soil Moisture from 2000-2018')
    plt.xlabel('Month')
    plt.ylabel('Average Soil Moisture')
    plt.xticks(rotation=45)
    plt.grid(False)  # Remove gridlines
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)  # Remove x-tick marks

    output_file_name = os.path.join(data_config['project_report_visuals'], 'bangladesh_monthly_soil_moisture.png')
    plt.savefig(output_file_name, bbox_inches='tight')
    plt.show()

    return monthly_averages


def plot_soil_moisture_grid(data_config_file_path: str) -> None:
    """
    Plot a grid of average monthly soil moisture images and save the plot as an image file. Each item in the grid represents a month.

    Args:
        data_config_file_path (str): Path to the data configuration JSON file containing the directory path for soil moisture images and the path for saving the plot.

    Returns:
        None: The function generates and saves a grid plot of monthly soil moisture images.

    """
    with open(data_config_file_path) as data_config_file:
        data_config = json.load(data_config_file)

    filepath = f"{data_config['project_report_visuals']}/soil_moisture"
    def extract_month_number(filename):
        return int(filename.split('_')[-1].split('.')[0])
    tif_files = sorted(os.listdir(filepath), key=extract_month_number)


    months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

    n_rows = 3
    n_cols = 4

    fig = plt.figure(figsize=(9, 10))

    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0)

    for i, (file, month) in enumerate(zip(tif_files, months)):
        ax = fig.add_subplot(gs[i])
        file = os.path.join(filepath, file)
        with rasterio.open(file) as src:
            data = src.read(1)
            im = ax.imshow(data, vmin=0, vmax=1, cmap='gray')
            ax.set_title(month, fontsize=10)
            ax.axis('off')  # Hide the axes ticks

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

    fig.suptitle("Average Monthly Soil Moisture as a Fraction", fontsize=15, y=0.92)
    # fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)
    plt.tight_layout()

    output_file_name = os.path.join(data_config['project_report_visuals'], 'bangladesh_soil_moisture_grid.png')
    plt.savefig(output_file_name, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    with open(os.environ["PROJECT_FLOOD_CORE_PATHS"]) as core_config_file:
        core_config = json.load(core_config_file)
    bangladesh_shape = generate_country_outline(core_config['bangladesh_shape_outline'])
    bangladesh_bounding_box_ee = ee.Geometry.BBox(*bangladesh_shape.bounds)


    # PLOTTING
    plot_bangladesh_flood_distribution(os.environ["PROJECT_FLOOD_CORE_PATHS"], os.environ["PROJECT_FLOOD_DATA_PATHS"])
    monthly_averages = plot_bangladesh_monthly_soil_moisture(os.environ["PROJECT_FLOOD_DATA_PATHS"], bangladesh_bounding_box_ee)
    plot_soil_moisture_grid(os.environ["PROJECT_FLOOD_DATA_PATHS"])