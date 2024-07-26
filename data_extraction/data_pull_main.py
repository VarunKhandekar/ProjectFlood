import ee
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from shapely.geometry import box
import warnings
from data_extraction.generic_helpers import *
from data_extraction.google_drive_helpers import *
from data_extraction.flood_image_helpers import *
from data_extraction.soil_moisture_helpers import *
from data_extraction.rainfall_helpers import * 

def process_flood_event(date, events, rainfall_days_before: int, rainfall_days_after: int, main_region_shape: ee.geometry.Geometry, rain_shape: Polygon, soil_moisture_shape: ee.geometry.Geometry, crs_transform_list):
    print("Running for flood event: ", date)
    gfd = ee.ImageCollection('GLOBAL_FLOOD_DB/MODIS_EVENTS/V1')
    perm_water = ee.ImageCollection('JRC/GSW1_4/MonthlyRecurrence')

    # Get flood events
    inundation_images = []
    for event in events:
        flood_image = ee.Image(gfd.filterMetadata('id', 'equals', int(event)).first()).select(['flooded'])
        inundation_images.append(flood_image)

    # Add permanent water overlay
    water_overlay = ee.Image(perm_water.filterMetadata('month', 'equals', date.month).first())
    occurrence_band = water_overlay.select('monthly_recurrence')
    water_overlay_filtered = occurrence_band.gt(perm_water_threshold)
    water_overlay_filtered = water_overlay_filtered.rename('flooded')
    water_overlay_filtered = water_overlay_filtered.setDefaultProjection("EPSG:4326", crs_transform_list)
    inundation_images.append(water_overlay_filtered.toByte())

    # Add sea overlay
    sea = ee.Image(ee.ImageCollection('MODIS/006/MOD44W').filterDate('2015-01-01', '2015-01-02').first()).select(['water_mask'])
    sea = sea.rename('flooded')
    sea = sea.setDefaultProjection("EPSG:4326", crs_transform_list)
    inundation_images.append(sea.toByte())

    # Export combo image to gdrive
    combo = ee.ImageCollection(inundation_images).reduce(ee.Reducer.max())
    export_task = ee.batch.Export.image.toDrive(
        image=combo,
        description=f'BangladeshWater{date.year}{date.month:02}{date.day:02}',
        folder='BangladeshFloodImages',
        fileNamePrefix=f'BangladeshWater{date.year}{date.month:02}{date.day:02}',
        region=main_region_shape,
        crs='EPSG:4326',
        # scale=250,  # Adjust the scale as needed
        crsTransform=crs_transform_list,
        maxPixels=1e13 
    )
    export_task.start()
    time.sleep(1) # sleep to ensure no duplication of EE request IDs

    # Pull soil moisture data for the day before, store to gdrive
    era5_soil_moisture = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    
    soil_moisture_date = pd.to_datetime(date) - timedelta(days=1)
    soil_moisture = ee.Image(era5_soil_moisture.filterDate(soil_moisture_date.strftime(r"%Y-%m-%d"), 
                                                           (soil_moisture_date+pd.Timedelta(days=1)).strftime(r'%Y-%m-%d')).first())
    soil_moisture = soil_moisture.select("volumetric_soil_water_layer_1")
    soil_moisture = soil_moisture.unmask(1)
    # soil_moisture = soil_moisture.resample(mode="bilinear")
    soil_moisture_task = ee.batch.Export.image.toDrive(
        image=soil_moisture,
        description=f'BangladeshSoilMoisture{soil_moisture_date.year}{soil_moisture_date.month:02}{soil_moisture_date.day:02}',
        folder='BangladeshSoilMoisture',
        fileNamePrefix=f'BangladeshSoilMoisture{soil_moisture_date.year}{soil_moisture_date.month:02}{soil_moisture_date.day:02}',
        region=soil_moisture_shape,
        crs='EPSG:4326',
        # scale=250,  # Adjust the scale as needed
        crsTransform=crs_transform_list,
        maxPixels=1e13 
    )
    soil_moisture_task.start()
    time.sleep(1) # sleep to ensure no duplication of EE request IDs

    # Pull rain data x days preceding, 1 day 'forecast'
    timestamps = generate_timestamps(date, rainfall_days_before, rainfall_days_after, freq="3h")
    files = generate_files_of_interest(drive, timestamps, 'static/config.json', 'MSWEP_Past_3hr_folder_id')
    assert len(timestamps) == len(files)

    for i, timestamp in enumerate(timestamps):
        rainfall_file = pull_and_crop_rainfall_data(drive, files[i], timestamp, "./data/temp/", rain_shape, 'static/config.json')
        send_to_google_drive(drive, rainfall_file, 'static/config.json', 'bangladesh_rainfall_folder_id', overwrite=True)




def process_non_flood_event(date, rainfall_days_before: int, rainfall_days_after: int, main_region_shape, rain_shape, soil_moisture_shape, crs_transform_list):
    print("Running for non-flood event: ", date)
    perm_water = ee.ImageCollection('JRC/GSW1_4/MonthlyRecurrence')

    water_images = []

    # Get permanent water
    water_overlay = ee.Image(perm_water.filterMetadata('month', 'equals', date.month).first())
    occurrence_band = water_overlay.select('monthly_recurrence')
    water_overlay_filtered = occurrence_band.gt(perm_water_threshold) 
    water_overlay_filtered = water_overlay_filtered.rename('flooded')
    water_overlay_filtered = water_overlay_filtered.setDefaultProjection("EPSG:4326", crs_transform_list)
    water_images.append(water_overlay_filtered.toByte())

    # Add sea overlay
    sea = ee.Image(ee.ImageCollection('MODIS/006/MOD44W').filterDate('2015-01-01', '2015-01-02').first()).select(['water_mask'])
    sea = sea.rename('flooded')
    sea = sea.setDefaultProjection("EPSG:4326", crs_transform_list)
    water_images.append(sea.toByte())

    # Export combo image to gdrive
    combo = ee.ImageCollection(water_images).reduce(ee.Reducer.max())
    export_task = ee.batch.Export.image.toDrive(
        image=combo,
        description=f'BangladeshWater{date.year}{date.month:02}{date.day:02}',
        folder='BangladeshNonFloodImages',
        fileNamePrefix=f'BangladeshWater{date.year}{date.month:02}{date.day:02}',
        region=main_region_shape,
        crs='EPSG:4326',
        # scale=250,  # Adjust the scale as needed
        crsTransform=crs_transform_list,
        maxPixels=1e13 
    )
    export_task.start()
    time.sleep(1) # sleep to ensure no duplication of EE request IDs

    # Get soil moisture data
    era5_soil_moisture = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')

    soil_moisture_date = pd.to_datetime(date) - timedelta(days=1)
    # print("soil_moisture: ", soil_moisture_date)
    soil_moisture = ee.Image(era5_soil_moisture.filterDate(soil_moisture_date.strftime(r"%Y-%m-%d"), 
                                                           (soil_moisture_date+pd.Timedelta(days=1)).strftime(r'%Y-%m-%d')).first())
    soil_moisture = soil_moisture.select("volumetric_soil_water_layer_1")
    soil_moisture = soil_moisture.unmask(1)
    # soil_moisture = soil_moisture.resample(mode="bilinear")
    soil_moisture_task = ee.batch.Export.image.toDrive(
        image=soil_moisture,
        description=f'BangladeshSoilMoisture{soil_moisture_date.year}{soil_moisture_date.month:02}{soil_moisture_date.day:02}',
        folder='BangladeshSoilMoistureNonFlood',
        fileNamePrefix=f'BangladeshSoilMoisture{soil_moisture_date.year}{soil_moisture_date.month:02}{soil_moisture_date.day:02}',
        region=soil_moisture_shape,
        crs='EPSG:4326',
        # scale=250,  # Adjust the scale as needed
        crsTransform=crs_transform_list,
        maxPixels=1e13 
    )
    soil_moisture_task.start()
    time.sleep(1) # sleep to ensure no duplication of EE request IDs
 
    # Get rain data
    timestamps = generate_timestamps(date, rainfall_days_before, rainfall_days_after, freq="3h")
    files = generate_files_of_interest(drive, timestamps, 'static/config.json', 'MSWEP_Past_3hr_folder_id')
    
    assert len(timestamps) == len(files)
    
    for i, timestamp in enumerate(timestamps):
        rainfall_file = pull_and_crop_rainfall_data(drive, files[i], timestamp, "./data/temp/", rain_shape, 'static/config.json')
        send_to_google_drive(drive, rainfall_file, 'static/config.json', 'bangladesh_rainfall_folder_id', overwrite=True)




if __name__=="__main__":
    # Suppress warnings about 'cfgrib' engine loading failure
    warnings.filterwarnings("ignore", message=".*Engine 'cfgrib' loading failed.*")

    start = time.time()
    # Set Up Google Earth Engine
    ee.Authenticate()
    with open('static/config.json') as config_file:
        config = json.load(config_file)
    earth_engine_project_name = config['earth_engine_project']
    ee.Initialize(project=earth_engine_project_name)
    print(ee.String('Hello from the Earth Engine servers!').getInfo())

    perm_water_threshold = 50 # x% of the time the area is covered in water

    crs_transform_list_high_res = [0.002245788210298804, 0.0, 88.08430518433968, 0.0, -0.002245788210298804, 26.44864775268901] #2043x2571
    # crs_transform_list = [0.0226492347869873, 0.0, 88.08430518433968, 0.0, -0.0226492347869873, 26.44864775268901] #203x256

    # Set Up Google Drive API
    # drive = authenticate("config.json")
    gauth = GoogleAuth()
    with open("config.json") as config_file:
        config = json.load(config_file)
    google_drive_credentials_path = config['google_drive_credentials']
    google_drive_oauth_path = config['google_drive_oauth']
    gauth.LoadClientConfigFile(google_drive_credentials_path)
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # Get Bangladesh outline
    bangladesh_shape = generate_country_outline('bangladesh-outline_68.geojson')
    bangladesh_bounding_box = box(*bangladesh_shape.bounds)
    bangladesh_bounding_box_ee = ee.Geometry.BBox(*bangladesh_shape.bounds)

    # Set up shapes
    main_region_shape = bangladesh_bounding_box_ee
    rain_shape = bangladesh_bounding_box
    soil_moisture_shape = bangladesh_bounding_box_ee

    # Rainfall days before and after
    rainfall_days_before = 7
    rainfall_days_after = 1

    # Max number of processes allowed to run
    max_workers = 6

    # TOPOLOGY DATA
    topology = ee.Image("NASA/NASADEM_HGT/001").select("elevation")

    # Replace all no data values with 0 (essentially the sea)
    topology = topology.unmask(0)

    export_task = ee.batch.Export.image.toDrive(
        image=topology,
        description='BangladeshTopology',
        folder='Topology',
        fileNamePrefix='BangladeshTopology',
        region=bangladesh_bounding_box_ee,
        crs='EPSG:4326',
        # scale=250,  # Adjust the scale as needed
        crsTransform=crs_transform_list_high_res,
        maxPixels=1e13  # Adjust maxPixels as needed
    )
    export_task.start()
    time.sleep(10)

    # FLOOD EVENTS
    refined_flood_events = generate_flood_events('Bangladesh_Flood_Events.xlsx')
 
    # Submit jobs, process pool to avoid memory issues
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_flood_event, date, events, rainfall_days_before, rainfall_days_after, main_region_shape, rain_shape, soil_moisture_shape) for date, events in refined_flood_events.items()]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                # print(f'Result: {result}')
            except Exception as exc:
                print(f'Generated an exception: {exc}')


    # NON-FLOOD EVENTS
    print("Running non flood events....")
    num_dates_to_generate = 100
    safety_window = 10
    random_dates = generate_random_non_flood_dates('Bangladesh_Flood_Events.xlsx', num_dates_to_generate, safety_window, "config.json")
    # random_dates = pd.read_csv("to_rerun.csv", header=None)
    # random_dates = list(random_dates.values.flatten())
    # random_dates = [pd.Timestamp(year=2000, month=9, day=18)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_non_flood_event, date, rainfall_days_before, rainfall_days_after, main_region_shape, rain_shape, soil_moisture_shape) for date in random_dates]

        for future in as_completed(futures):
            try:
                result = future.result()
                # print(f'Result: {result}')
            except Exception as exc:
                print(f'Generated an exception: {exc}')

    
    end = time.time()
    print(end - start)