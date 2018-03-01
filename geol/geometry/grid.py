"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>
import geopandas as gpd
from geol.utils import constants
import six
import abc
import os
import json
import fiona
import errno


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

    def to_shp(self, out):
        self._grid.to_file(os.path.abspath(out), driver='ESRI Shapefile')

    def to_geojson(self, out):
        self._grid.reset_index().rename(
            columns={"index": "cellID"}).to_file(out, driver='GeoJSON')

    def assign_crs(self, crs):
        if crs is not None:
            self._grid.to_crs({'init': crs})

    def silentremove(self, filename):
        try:
            os.remove(filename)
        except OSError as e:  # this would be "except OSError, e:" before Python 2.6
            if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
                raise  # re-raise exception if a different error occurred

    def write(self, out, driver="ESRI Shapefile", crs=None, schema=None):
        # be sure to assign a CRS
        self.assign_crs(crs)
        # check if file exists and remove it
        # see https://github.com/Toblerity/Fiona/pull/532
        self.silentremove(out)
        # save in the requested format
        if driver == 'GeoJSON':
            self.to_geojson(out)
        else:
            self.to_shp(out)

    @property
    def grid(self):
        return self._grid
