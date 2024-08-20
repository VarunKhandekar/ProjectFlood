import ee
import time
from shapely.geometry import box
from data_extraction.generic_helpers import *
from data_extraction.flood_image_helpers import *


if __name__=="__main__":
    # Set Up Google Earth Engine
    ee.Authenticate()
    with open(os.environ["PROJECT_FLOOD_CORE_PATHS"]) as core_config_file:
        core_config = json.load(core_config_file)
    earth_engine_project_name = core_config['earth_engine_project']
    ee.Initialize(project=earth_engine_project_name)
    print(ee.String('Hello from the Earth Engine servers!').getInfo())

    bangladesh_shape = generate_country_outline(core_config['bangladesh_shape_outline'])
    bangladesh_bounding_box = box(*bangladesh_shape.bounds)
    bangladesh_bounding_box_ee = ee.Geometry.BBox(*bangladesh_shape.bounds)

    crs_transform_list_high_res = [0.002245788210298804, 0.0, 88.08430518433968, 0.0, -0.002245788210298804, 26.44864775268901]

    # Loop through all images in non_flood and flood
    with open(os.environ["PROJECT_FLOOD_DATA"]) as data_config_file:
        data_config = json.load(data_config_file)

    # Go through flood and non-flood images.
    water_images = []
    water_images.extend(os.listdir(data_config["non_flood_file_path_2044_2573"]))
    water_images.extend(os.listdir(data_config["flood_file_path_2044_2573"]))

    perm_water = ee.ImageCollection('JRC/GSW1_4/MonthlyRecurrence')
    era5_soil_moisture = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    for i in os.listdir(data_config["non_flood_file_path_2044_2573"]):
        date = pd.to_datetime(i[-12:-4], format=r'%Y%m%d')
        soil_moisture_date = date - pd.Timedelta(days=1)

        water_overlay = ee.Image(perm_water.filterMetadata('month', 'equals', date.month).first())
        occurrence_band = water_overlay.select('monthly_recurrence')
        soil_moisture = ee.Image(era5_soil_moisture.filterDate(soil_moisture_date.strftime(r"%Y-%m-%d"), 
                                                           (soil_moisture_date+pd.Timedelta(days=1)).strftime(r'%Y-%m-%d')).first())
        soil_moisture = soil_moisture.select("volumetric_soil_water_layer_1")

        sea_water_proxy = occurrence_band.gt(0.99)
        masked_soil_moisture = soil_moisture.updateMask(sea_water_proxy.eq(1))

        # Reduce the masked soil moisture image to get the average soil moisture value
        average_soil_moisture = masked_soil_moisture.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=bangladesh_bounding_box_ee,
            crs='EPSG:4326',
            crsTransform=crs_transform_list_high_res
        ).getInfo()
        masked_soil_moisture = soil_moisture.updateMask(sea_water_proxy.eq(1))

        soil_moisture = soil_moisture.unmask(average_soil_moisture["volumetric_soil_water_layer_1"])

        soil_moisture_task = ee.batch.Export.image.toDrive(
            image=soil_moisture,
            description=f'BangladeshSoilMoisture{soil_moisture_date.year}{soil_moisture_date.month:02}{soil_moisture_date.day:02}',
            folder='BangladeshSoilMoistureNonFlood_watermask',
            fileNamePrefix=f'BangladeshSoilMoisture{soil_moisture_date.year}{soil_moisture_date.month:02}{soil_moisture_date.day:02}',
            region=bangladesh_bounding_box_ee,
            crs='EPSG:4326',
            # scale=250,  # Adjust the scale as needed
            crsTransform=crs_transform_list_high_res,
            maxPixels=1e13 
        )
        soil_moisture_task.start()
        time.sleep(1) # sleep to ensure no duplication of EE request IDs
    
    for i in os.listdir(data_config["flood_file_path_2044_2573"]):
        date = pd.to_datetime(i[-12:-4], format=r'%Y%m%d')
        soil_moisture_date = date - pd.Timedelta(days=1)
        
        water_overlay = ee.Image(perm_water.filterMetadata('month', 'equals', date.month).first())
        occurrence_band = water_overlay.select('monthly_recurrence')
        soil_moisture = ee.Image(era5_soil_moisture.filterDate(soil_moisture_date.strftime(r"%Y-%m-%d"), 
                                                           (soil_moisture_date+pd.Timedelta(days=1)).strftime(r'%Y-%m-%d')).first())
        soil_moisture = soil_moisture.select("volumetric_soil_water_layer_1")

        sea_water_proxy = occurrence_band.gt(0.99)
        masked_soil_moisture = soil_moisture.updateMask(sea_water_proxy.eq(1))

        # Reduce the masked soil moisture image to get the average soil moisture value
        average_soil_moisture = masked_soil_moisture.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=bangladesh_bounding_box_ee,
            crs='EPSG:4326',
            crsTransform=crs_transform_list_high_res
        ).getInfo()
        masked_soil_moisture = soil_moisture.updateMask(sea_water_proxy.eq(1))

        soil_moisture = soil_moisture.unmask(average_soil_moisture["volumetric_soil_water_layer_1"])

        soil_moisture_task = ee.batch.Export.image.toDrive(
            image=soil_moisture,
            description=f'BangladeshSoilMoisture{soil_moisture_date.year}{soil_moisture_date.month:02}{soil_moisture_date.day:02}',
            folder='BangladeshSoilMoistureFlood_watermask',
            fileNamePrefix=f'BangladeshSoilMoisture{soil_moisture_date.year}{soil_moisture_date.month:02}{soil_moisture_date.day:02}',
            region=bangladesh_bounding_box_ee,
            crs='EPSG:4326',
            # scale=250,  # Adjust the scale as needed
            crsTransform=crs_transform_list_high_res,
            maxPixels=1e13 
        )
        soil_moisture_task.start()
        time.sleep(1) # sleep to ensure no duplication of EE request IDs

    