"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

from geol.geometry.grid import Grid
from geol.utils import utils, constants
import geopandas as gpd
import math
from shapely.geometry import Polygon
import os

class SquareGrid(Grid):

    def __init__(self, base_shape, meters=50, window_size=None, crs=constants.universal_crs):

        super().__init__(crs=crs)

        # Compute Bounding Box if requested
        if self.__window_size is not None:
            self.__base_shape = utils.build_bbox(self.__base_shape)
        else:
            self.__base_shape = base_shape

        self.__meters = meters
        self.__window_size = window_size

        self.__build()

    @classmethod
    def from_file(cls, filepath, meters=50, window_size=None, input_crs=constants.default_crs):
        base_shape = gpd.GeoDataFrame.from_file(os.path.abspath(filepath))
        base_shape.crs = {'init':input_crs}
        cls(base_shape, meters, window_size)

    @classmethod
    def from_name(cls, area_name, meters=50, window_size=None):
        base_shape = utils.get_area_boundary(area_name)
        cls(base_shape, meters, window_size)


    def __build(self):

        # Re-project data
        self.logger.debug("Convert area to crs epsg:" + constants.universal_crs + ".")

        # We work with the universal crs epsg:3857
        area = self.__base_shape.to_crs({'init': 'epsg:' + constants.universal_crs, 'units': 'm'})

        self.logger.debug("Defining boundaries.")

        # Obtain the boundaries of the geometry
        boundaries = dict({'min_x': area.total_bounds[0],
                           'min_y': area.total_bounds[1],
                           'max_x': area.total_bounds[2],
                           'max_y': area.total_bounds[3]})

        # Find number of square for each side
        x_squares = int(math.ceil(math.fabs(boundaries['max_x'] - boundaries['min_x']) / self.__meters))
        y_squares = int(math.ceil(math.fabs(boundaries['min_y'] - boundaries['max_y']) / self.__meters))

        # placeholder for the polygon
        polygons = []

        shape = area.unary_union

        self.logger.debug("Creating cells.")
        # iterate on the x
        for i in range(0, x_squares):

            # increment x
            x1 = boundaries['min_x'] + (self.__meters * i)
            x2 = boundaries['min_x'] + (self.__meters * (i + 1))

            # iterate on y
            for j in range(0, y_squares):
                # increment y
                y1 = boundaries['min_y'] + (self.__meters * j)
                y2 = boundaries['min_y'] + (self.__meters * (j + 1))
                polygon_desc = {}

                # Create shape (polygon)
                p = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])

                # Compute intersection between boros and the current cell and check if it's true
                # ALTERNATIVELY COMPUTE CENTROID AND THEN THE INTERSECTION WITH
                # THE BOROS AND CHECK IF IT IS TRUE

                # s = boros_shape.intersection(p)
                s = shape.intersects(p)

                # if(s.area>0):
                if (s):
                    # set polygon information
                    polygon_desc['id_x'] = i
                    polygon_desc['id_y'] = j
                    # shape.intersection(p) ATTENTION! If you use the
                    # intersection than the crawler fails!
                    polygon_desc['geometry'] = p
                    polygons.append(polygon_desc)

        self.logger.debug("End creation of cells.")

        # Create the geoDataFrame and convert to the input crs.
        gdf = gpd.GeoDataFrame(polygons, crs={'init': 'epsg:' + constants.universal_crs, 'units': 'm'})
        self.__grid = gdf.to_crs({'init': self.__crs})

    @property
    def base_shape(self):
        return self.__base_shape

    @property
    def window_size(self):
        return self.__window_size

    @property
    def meters(self):
        return self.__meters