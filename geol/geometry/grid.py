"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>
import geopandas as gpd
from geol.utils import constants
import six
import abc
import os

@six.add_metaclass(abc.ABCMeta)
class Grid:

    def __init__(self, crs):

        self.__grid = None
        self.__crs = crs

    @classmethod
    def from_file(cls, inputfile, crs=constants.default_crs):
        base_shape = gpd.GeoDataFrame.from_file(os.path.abspath(inputfile))
        base_shape.crs = {'init': crs}

    @abc.abstractmethod
    def __build(self):
        pass

    def write(self, filename, outputpath="./"):

        #gdf.sort_values(by=["id_x", "id_y"], ascending=True, inplace=True)
        self.__gird.reset_index().rename(columns={"index": "cellID"})\
            .to_file(os.path.abspath(outputpath + filename), driver='ESRI Shapefile')

    @property
    def grid(self):
           return self.__grid