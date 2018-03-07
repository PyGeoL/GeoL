
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
import os
import re
import zipfile,fnmatch
import shutil

import sys

# SET CITY NAME
CITIES = ["barcelona","london","rome","milan","madrid","berlin","paris","stockholm"]
# Base directory
BASE_DIR = os.path.abspath(".")
# base directory for data files
BASE_DIR_DATA = os.path.join(BASE_DIR, "data")
directoriesToBuild = ["grid"] #'clipped','mapped','grid','foursquare_raw','train','dev','test','count','landuse','embeddings','osm_raw]

# build folder structure for each city
def makeDirStruct(city):
    for directory in directoriesToBuild:
        if not os.path.exists(os.path.join(BASE_DIR_DATA,city,directory)):
            print(os.path.join(BASE_DIR_DATA,city,directory))
#             shutil.rmtree(os.path.join(BASE_DIR_DATA,city,directory))
            os.makedirs(os.path.join(BASE_DIR_DATA,city,directory))

# create directories and unzip files

for CITY_NAME in CITIES:
    makeDirStruct(CITY_NAME)


