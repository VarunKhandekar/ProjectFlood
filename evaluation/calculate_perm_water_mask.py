from visualisations.visualisation_helpers import *
import os
import numpy as np


if __name__ == "__main__":
    resolution = 256
    with open(os.environ["PROJECT_FLOOD_DATA"]) as data_config_file:
        data_config = json.load(data_config_file)

    water_image_path = f"{data_config['water_image_path']}_{resolution}_{resolution}"

    water_images = os.listdir(water_image_path)
    combined_mask = strip_black_pixel_padding_PIL(os.environ["PROJECT_FLOOD_CORE_PATHS"], resolution, os.path.join(water_image_path, water_images[0]))

    for water_image in os.listdir(water_image_path)[1:]:
        cropped_image = strip_black_pixel_padding_PIL(os.environ["PROJECT_FLOOD_CORE_PATHS"], resolution, os.path.join(water_image_path, water_image))
        combined_mask = np.logical_and(combined_mask, cropped_image)
        
    print(combined_mask)
    np.save(f"{data_config['model_results_path']}/perm_water_mask_{resolution}.npy", combined_mask)
