"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

import geopandas as gpd
from ..factory import base
from ..utils import constants, utils
import math
import re
import shapely
import logging
import os
import geopandas


class Square(base.TessellationFactory):
    """
    Factory class to build squared tessellation.
    """

    def __init__(self, meters=50, crs=constants.default_crs,
                 area=None, area_name=None, which_result=None, logger=None, extension="boundary"):
        """__init__ method to instantiate variables.

        Args:
            area: polygon (shape) that it is used as a base to build the geometry. In case input is not None, base_shape
                            should indicate the shape over the which the grid has been built.
            area_name: name of the area you want to build the grid over. It has to be in the OSM format like area name, country
            input: load grid from the input path
            meters: size of the square
            crs: coordinate reference system of the output grid
        """

        if logger is None:
            logging.basicConfig(format='%(levelname)s: %(message)s')
            self.logger = logging.getLogger(__name__)

        self.__area = area
        self.__area_name = area_name
        self.__which_result = which_result
        self.__meters = meters
        self.__crs = crs
        self.__extension = extension

    def set_properties(self):

        if ((self.__area is None) and (self.__area_name is None)):
            raise ValueError("Arguments are mutually exclusive. To create a Square you must indicate an area or the " +
                             "path to the shape.")

        regex = re.compile('[^a-zA-Z]')

        prop = {"area_name": self.__area_name, "meters": self.__meters, "crs:": self.__crs,
                "id": regex.sub("", self.__area_name) + "-Square-" + str(self.__meters)}

        return prop

    # TODO broken
    def get_properties(self):
        return self.__prop

    def build_bbox(self, area, bbox_side_len=500):
        """

        :param area: area whose centroid is used as a starting point for building the tessellation
        :param bbox_side_len: length of the bbox rectangle. Defaults to 500 meters.

        """

       # get area centroid
        centroid = area.centroid[0].coords[0]

        # get North-East corner
        NE = [float(coord)+bbox_side_len for coord in centroid]
        # get South-West corner
        SW = [float(coord)-bbox_side_len for coord in centroid]

        # build bbox from NE,SW corners
        bbox = shapely.geometry.box(SW[0], SW[1], NE[0], NE[1], ccw=True)
        poly_df = gpd.GeoDataFrame(geometry=[bbox])
        print(type(poly_df))
        poly_df.crs = {'init': 'epsg:' + constants.universal_crs, 'units': 'm'}
        return poly_df

    def build_tessellation(self):

        self.logger.debug("Getting/Loading area shape.")
        if self.__area is not None:
            # read the shape file containing the boundaries of the area.
            area = gpd.read_file(self.__area)
        elif self.__area_name is not None:
            area = utils.get_area_boundary(
                self.__area_name, self.__which_result)

        # Re-project data
        self.logger.debug("Convert area to crs epsg:" +
                          constants.universal_crs + ".")
        # We work with the universal crs epsg:3857
        area = area.to_crs(
            {'init': 'epsg:' + constants.universal_crs, 'units': 'm'})

        # Compute Bounding Box if requested
        if self.__extension == 'bbox':
            # TODO ugly owerwriting. To be refactored.
            area = self.build_bbox(area)

        self.logger.debug("Defining boundaries.")
        # Obtain the boundaries of the geometry
        boundaries = dict({'min_x': area.total_bounds[0],
                           'min_y': area.total_bounds[1],
                           'max_x': area.total_bounds[2],
                           'max_y': area.total_bounds[3]})

        # Find number of square for each side
        x_squares = int(math.ceil(
            math.fabs(boundaries['max_x'] - boundaries['min_x']) / self.__meters))
        y_squares = int(math.ceil(
            math.fabs(boundaries['min_y'] - boundaries['max_y']) / self.__meters))

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
                p = shapely.geometry.Polygon(
                    [(x1, y1), (x1, y2), (x2, y2), (x2, y1)])

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
        gdf = gpd.GeoDataFrame(
            polygons, crs={'init': 'epsg:' + constants.universal_crs, 'units': 'm'})
        gdf = gdf.to_crs({'init': self.__crs})
        gdf.sort_values(by=["id_x", "id_y"], ascending=True, inplace=True)
        gdf = gdf.reset_index().rename(columns={"index": "cellID"})

        return gdf, area
