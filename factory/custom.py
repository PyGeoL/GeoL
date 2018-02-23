"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

import geopandas as gpd
from ..factory import base
from ..utils import constants
import time
import re

class Custom(base.TessellationFactory):

    def __init__(self, input=None, crs=constants.default_crs, area_name=None):

        self.__input = input
        self.__crs = crs
        self.__area_name = area_name

    def build_tessellation(self):

        """
         TODO: il logger non funziona cosi'! Va settato che funzioni su multipli file.
         TODO: add the creation of the area building the bounding box that contains the tessellation.
         """

        self.logger.info("Start loading tessellation (" + self.__properties['id'] + ")")
        start = time.time()

        tessellation = self.read(input)

        end = time.time()
        self.logger.info("End loading grid (" + self.__properties['id'] + ") in " + str(end - start))

        return tessellation, None

    def read(self, input, crs=constants.default_crs):

        tessellation = gpd.read_file(input)

        # By default, we assume that the grid has crs=epsg:4326
        tessellation.crs = {'init': crs}

        if "id" in tessellation.columns:
            tessellation.drop("id", axis=1, inplace=True)

        return tessellation

    def set_properties(self):

        if self.__input is None:
            raise ValueError("Input must be set.")
        elif self.__area_name is None:
            raise ValueError("Area name must be set.")

        regex = re.compile('[^a-zA-Z]')

        prop = {"area_name": self.__area_name,
                "crs:": self.__crs,
                "id": regex.sub("", self.__area_name)}

        return prop