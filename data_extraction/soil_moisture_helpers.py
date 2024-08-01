# import cdsapi
# from shapely.geometry import mapping
# import zipfile
# import xarray as xr
# import os

# def pull_soil_moisture_copernicus(soil_moisture_date, output_path: str, shape) -> str:
#     """
#     Retrieve and process soil moisture data from Copernicus.

#     Args:
#         soil_moisture_date (pd.Timestamp): The date for which to retrieve soil moisture data.
#         output_path (str): The directory where the processed data will be saved.
#         shape (shapely.geometry.Polygon): A shapely Polygon object defining the area to clip the data to.

#     Returns:
#         str: The path to the processed NetCDF file.

#     This function performs the following steps:
#         1. Sets up and retrieves soil moisture data from the Copernicus Climate Data Store in ZIP format.
#         2. Extracts the ZIP file to obtain the NetCDF (.nc) file.
#         3. Renames the extracted file and deletes the original ZIP file.
#         4. Clips the data to the specified geographic shape and saves the clipped data as a new NetCDF file.
#     """
#     # Set up Copernicus
#     c = cdsapi.Client()

#     zip_path = f'data/soil_moisture_bangladesh_{soil_moisture_date.year}{soil_moisture_date.month:02}{soil_moisture_date.day:02}.zip'
#     # with contextlib.redirect_stdout(None):
#     c.retrieve(
#     'satellite-soil-moisture',
#     {
#         'format': 'zip',
#         'year': f'{soil_moisture_date.year}',
#         'day': f'{soil_moisture_date.day:02}',
#         'variable': 'surface_soil_moisture',
#         'type_of_sensor': 'active',
#         'time_aggregation': 'day_average',
#         'month': f'{soil_moisture_date.month:02}',
#         'type_of_record': 'cdr', #may need to change to icdr when considering going 'live'
#         'version': 'v202212',
#     },
#     zip_path
#     )

#     new_path = os.path.join(output_path, f'soil_moisture_bangladesh_{soil_moisture_date.year}{soil_moisture_date.month:02}{soil_moisture_date.day:02}.nc')
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         file_list = zip_ref.namelist()
#         zip_ref.extract(file_list[0], path=output_path)       
#         os.rename(os.path.join(output_path, file_list[0]), new_path)
#         # print(f'File extracted and renamed to: {new_path}')
#     os.remove(zip_path)

#     # Clip the data and save down
#     geojson = [mapping(shape)] # Use GeoJSON of the shape
#     with xr.open_dataset(new_path, engine="netcdf4") as data:
#         data = data['sm']
#         data.rio.set_spatial_dims('lon', 'lat', inplace=True)
#         data.rio.write_crs("EPSG:4326", inplace=True)
        
#         data_clipped = data.rio.clip(geojson, all_touched=True)
#     data_clipped.to_netcdf(new_path, mode='w')

#     return new_path