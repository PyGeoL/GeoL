"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>
import geopandas as gpd
from geol.utils import constants, utils
import six
import abc
import os
import json


@six.add_metaclass(abc.ABCMeta)
class Grid:

    def __init__(self, crs, grid=None):

        self._grid = grid
        self._crs = crs

    @classmethod
    def from_file(cls, inputfile, crs=constants.default_crs):
        grid = gpd.GeoDataFrame.from_file(os.path.abspath(inputfile))
        grid.crs = {'init': crs}
        return cls(crs, grid=grid)

    def write(self, out, driver="ESRI Shapefile", crs=None, schema=None):
        # be sure to assign a CRS
        utils.assign_crs(self._grid, crs)
        # check if file exists and remove it
        # see https://github.com/Toblerity/Fiona/pull/532
        utils.silentremove(out)
        # save in the requested format
        if driver == 'GeoJSON':
            self._grid.reset_index().rename(
                columns={"index": "cellID"}).to_file(out, driver='GeoJSON')
        else:
            self._grid.to_file(os.path.abspath(out), driver='ESRI Shapefile')

    @property
    def grid(self):
        return self._grid
