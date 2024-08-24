from visualisations.visualisation_helpers import *
import os
import numpy as np


if __name__ == "__main__":
    resolution = 256
    with open(os.environ["PROJECT_FLOOD_DATA"]) as data_config_file:
        data_config = json.load(data_config_file)

    water_image_path = f"{data_config['data_config']}_{resolution}_{resolution}"

    image_paths = os.listdir(water_image_path)
    combined_mask = strip_black_pixel_padding_PIL(os.environ["PROJECT_FLOOD_CORE_PATHS"], resolution, image_paths[0])

    for i in os.listdir(water_image_path)[1:]:
        cropped_image = strip_black_pixel_padding_PIL(os.environ["PROJECT_FLOOD_CORE_PATHS"], resolution, i)
        combined_mask = np.logical_and(combined_mask, cropped_image)

        np.save('perm_water_mask.npy', combined_mask)

    