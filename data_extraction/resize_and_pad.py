import os
from data_extraction.generic_helpers import *
from data_extraction.rainfall_helpers import *


if __name__ == "__main__":
    os.getcwd()
    resize_and_pad_file_paths = ["/home/vkhandekar/project_flood/BangladeshFloodImages_raw",
                                 "/home/vkhandekar/project_flood/BangladeshNonFloodImages_raw",
                                 "/home/vkhandekar/project_flood/Topology_raw",
                                 "/home/vkhandekar/project_flood/BangladeshSoilMoisture_raw",
                                 "/home/vkhandekar/project_flood/BangladeshSoilMoistureNonFlood_raw"]
    
    for fp in resize_and_pad_file_paths:
        target_file_path = "xx"
        for f in os.listdir(fp):
            pass