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

        self._grid = None
        self._crs = crs

    @classmethod
    def from_file(cls, inputfile, crs=constants.default_crs):
        base_shape = gpd.GeoDataFrame.from_file(os.path.abspath(inputfile))
        base_shape.crs = {'init': crs}

    #@abc.abstractmethod
    #def __build(self):
    #    pass

    def write(self, out, driver="ESRI Shapefile", crs=None):

        if crs is not None:
            tmp = self._grid.to_crs({'init': crs})
            tmp.reset_index().rename(columns={"index": "cellID"}).to_file(os.path.abspath(out), driver='ESRI Shapefile')
        else:
            #gdf.sort_values(by=["id_x", "id_y"], ascending=True, inplace=True)
            self._grid.reset_index().rename(columns={"index": "cellID"})\
                .to_file(out, driver='ESRI Shapefile')

        #TODO ADD SAVE GRID IN GEOJSON

    @property
    def grid(self):
           return self._grid