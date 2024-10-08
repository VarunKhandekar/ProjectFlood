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

def process_flood_event(date: pd.Timestamp, events: list, rainfall_days_before: int, rainfall_days_after: int, 
                        main_region_shape: ee.geometry.Geometry, rain_shape: Polygon, soil_moisture_shape: ee.geometry.Geometry, crs_transform_list: list) -> None:
    """
    Gather flood, soil moisture, and rainfall data, for a list of overlapping flood events. Export the results to Google Drive.

    Args:
        date (datetime-like): The date of the flood event.
        events (list): List of event IDs to retrieve flood images from the Global Flood Database.
        rainfall_days_before (int): Number of days before the flood event to include in the rainfall data.
        rainfall_days_after (int): Number of days after the flood event to include in the rainfall data.
        main_region_shape (ee.geometry.Geometry): Geometry for the region of interest to extract flood images.
        rain_shape (Polygon): The polygon shape for cropping the rainfall data.
        soil_moisture_shape (ee.geometry.Geometry): Geometry for the region to extract soil moisture data.
        crs_transform_list (list): Coordinate reference system (CRS) transform parameters for geospatial data.
    
    Returns:
        None

    """
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
    print(export_task.id)
    time.sleep(1) # sleep to ensure no duplication of EE request IDs

    # Pull soil moisture data for the day before, store to gdrive
    era5_soil_moisture = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    
    soil_moisture_date = pd.to_datetime(date) - timedelta(days=1)
    soil_moisture = ee.Image(era5_soil_moisture.filterDate(soil_moisture_date.strftime(r"%Y-%m-%d"), 
                                                           (soil_moisture_date+pd.Timedelta(days=1)).strftime(r'%Y-%m-%d')).first())
    soil_moisture = soil_moisture.select("volumetric_soil_water_layer_1")

    sea_water_proxy = occurrence_band.gt(0.99)
    masked_soil_moisture = soil_moisture.updateMask(sea_water_proxy.eq(1))

    # Reduce the masked soil moisture image to get the average soil moisture value
    average_soil_moisture = masked_soil_moisture.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=main_region_shape,
        crs='EPSG:4326',
        crsTransform=crs_transform_list
    ).getInfo()

    # soil_moisture = soil_moisture.unmask(1)
    soil_moisture = soil_moisture.unmask(average_soil_moisture["volumetric_soil_water_layer_1"])
    # soil_moisture = soil_moisture.unmask(1)
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
    files = generate_files_of_interest(drive, timestamps, os.environ["PROJECT_FLOOD_CORE_PATHS"], 'MSWEP_Past_3hr_folder_id')
    assert len(timestamps) == len(files)

    for i, timestamp in enumerate(timestamps):
        rainfall_file = pull_and_crop_rainfall_data(drive, files[i], timestamp, os.environ["PROJECT_FLOOD_DATA"], rain_shape, os.environ["PROJECT_FLOOD_CORE_PATHS"])
        send_to_google_drive(drive, rainfall_file, os.environ["PROJECT_FLOOD_CORE_PATHS"], 'bangladesh_rainfall_folder_id', overwrite=True)




def process_non_flood_event(date: pd.Timestamp, rainfall_days_before: int, rainfall_days_after: int, 
                            main_region_shape: ee.geometry.Geometry, rain_shape: Polygon, soil_moisture_shape: ee.geometry.Geometry, crs_transform_list: list) -> None:
    """
    Process a non-flood event by retrieving water, soil moisture, and rainfall data. Export the results to Google Drive.

    Args:
        date (pd.Timestamp): The date of the non-flood event.
        rainfall_days_before (int): Number of days before the event to include in the rainfall data.
        rainfall_days_after (int): Number of days after the event to include in the rainfall data.
        main_region_shape (ee.geometry.Geometry): Geometry for the region of interest to extract water images.
        rain_shape (Polygon): The polygon shape for cropping the rainfall data.
        soil_moisture_shape (ee.geometry.Geometry): Geometry for the region to extract soil moisture data.
        crs_transform_list (list): Coordinate reference system (CRS) transform parameters for geospatial data.
    
    Returns:
        None

    """
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
        folder='BangladeshNonFloodImages2',
        fileNamePrefix=f'BangladeshWater{date.year}{date.month:02}{date.day:02}',
        region=main_region_shape,
        crs='EPSG:4326',
        # scale=250,  # Adjust the scale as needed
        crsTransform=crs_transform_list,
        maxPixels=1e13 
    )
    export_task.start()
    print(export_task.id)
    time.sleep(1) # sleep to ensure no duplication of EE request IDs

    # Get soil moisture data
    era5_soil_moisture = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')

    soil_moisture_date = pd.to_datetime(date) - timedelta(days=1)
    # print("soil_moisture: ", soil_moisture_date)
    soil_moisture = ee.Image(era5_soil_moisture.filterDate(soil_moisture_date.strftime(r"%Y-%m-%d"), 
                                                           (soil_moisture_date+pd.Timedelta(days=1)).strftime(r'%Y-%m-%d')).first())
    soil_moisture = soil_moisture.select("volumetric_soil_water_layer_1")

    # sea_water_proxy = occurrence_band.gt(0.99)
    # masked_soil_moisture = soil_moisture.updateMask(sea_water_proxy.eq(1))

    # # Reduce the masked soil moisture image to get the average soil moisture value
    # average_soil_moisture = masked_soil_moisture.reduceRegion(
    #     reducer=ee.Reducer.mean(),
    #     geometry=main_region_shape,
    #     crs='EPSG:4326',
    #     crsTransform=crs_transform_list
    # ).getInfo()

    soil_moisture = soil_moisture.unmask(1)
    # soil_moisture = soil_moisture.unmask(average_soil_moisture["volumetric_soil_water_layer_1"])
    # soil_moisture = soil_moisture.resample(mode="bilinear")
    soil_moisture_task = ee.batch.Export.image.toDrive(
        image=soil_moisture,
        description=f'BangladeshSoilMoisture{soil_moisture_date.year}{soil_moisture_date.month:02}{soil_moisture_date.day:02}',
        folder='BangladeshSoilMoistureNonFlood2',
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
    files = generate_files_of_interest(drive, timestamps, os.environ["PROJECT_FLOOD_CORE_PATHS"], 'MSWEP_Past_3hr_folder_id')
    
    assert len(timestamps) == len(files)
    
    for i, timestamp in enumerate(timestamps):
        rainfall_file = pull_and_crop_rainfall_data(drive, files[i], timestamp, os.environ["PROJECT_FLOOD_DATA"], rain_shape, os.environ["PROJECT_FLOOD_CORE_PATHS"], 256)
        send_to_google_drive(drive, rainfall_file, os.environ["PROJECT_FLOOD_CORE_PATHS"], 'bangladesh_rainfall_folder_id', overwrite=True)


if __name__=="__main__":
    # Suppress warnings about 'cfgrib' engine loading failure
    warnings.filterwarnings("ignore", message=".*Engine 'cfgrib' loading failed.*")

    start = time.time()
    # Set Up Google Earth Engine
    ee.Authenticate()
    with open(os.environ["PROJECT_FLOOD_CORE_PATHS"]) as core_config_file:
        core_config = json.load(core_config_file)
    earth_engine_project_name = core_config['earth_engine_project']
    ee.Initialize(project=earth_engine_project_name)
    print(ee.String('Hello from the Earth Engine servers!').getInfo())

    perm_water_threshold = 50 # x% of the time the area is covered in water

    crs_transform_list_high_res = [0.002245788210298804, 0.0, 88.08430518433968, 0.0, -0.002245788210298804, 26.44864775268901] #2043x2571 - 250m approx
    # crs_transform_list = [0.0226492347869873, 0.0, 88.08430518433968, 0.0, -0.0226492347869873, 26.44864775268901] #203x256

    # Set Up Google Drive API
    # drive = authenticate("config.json")
    gauth = GoogleAuth()
    google_drive_credentials_path = core_config['google_drive_credentials']
    google_drive_oauth_path = core_config['google_drive_oauth']
    gauth.LoadClientConfigFile(google_drive_credentials_path)
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # Get Bangladesh outline
    bangladesh_shape = generate_country_outline(core_config['bangladesh_shape_outline'])
    bangladesh_bounding_box = box(*bangladesh_shape.bounds)
    bangladesh_bounding_box_ee = ee.Geometry.BBox(*bangladesh_shape.bounds)
    bangladesh_bounding_box_square_ee = ee.Geometry.BBox(*generate_square_coordinates_from_polygon(bangladesh_shape))

    # Set up shapes
    main_region_shape = bangladesh_bounding_box_ee
    rain_shape = bangladesh_bounding_box
    soil_moisture_shape = bangladesh_bounding_box_ee

    # Rainfall days before and after
    rainfall_days_before = 7
    rainfall_days_after = 1

    # Max number of processes allowed to run
    max_workers = 6

    # TOPOGRAPHY DATA
    topography = ee.Image("NASA/NASADEM_HGT/001").select("elevation")

    # Replace all no data values with 0 (essentially the sea)
    topography = topography.unmask(0)

    export_task = ee.batch.Export.image.toDrive(
        image=topography,
        description='BangladeshTopography',
        folder='Topography',
        fileNamePrefix='BangladeshTopography',
        region=bangladesh_bounding_box_ee,
        crs='EPSG:4326',
        # scale=250,  # Adjust the scale as needed
        crsTransform=crs_transform_list_high_res,
        maxPixels=1e13  # Adjust maxPixels as needed
    )
    export_task.start()
    time.sleep(10)

    # FLOOD EVENTS
    refined_flood_events = generate_flood_events(os.environ["PROJECT_FLOOD_CORE_PATHS"])
    store_flood_dates(refined_flood_events) #Save list of flood dates under consideration
 
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
    num_dates_to_generate = 250
    safety_window = 10
    random_dates = generate_random_non_flood_dates(os.environ["PROJECT_FLOOD_CORE_PATHS"], num_dates_to_generate, safety_window, os.environ["PROJECT_FLOOD_DATA"])

    # random_dates = pd.read_csv(core_config['rerun_path'], header=None)
    # random_dates = list(random_dates.values.flatten())
    # random_dates = [pd.to_datetime(i) for i in random_dates]

    # random_dates = [pd.Timestamp(year=2000, month=9, day=18)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_non_flood_event, date, rainfall_days_before, rainfall_days_after, main_region_shape, rain_shape, soil_moisture_shape, crs_transform_list_high_res) for date in random_dates]

        for future in as_completed(futures):
            try:
                result = future.result()
                # print(f'Result: {result}')
            except Exception as exc:
                print(f'Generated an exception: {exc}')

    
    end = time.time()
    print(end - start)