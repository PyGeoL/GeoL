"""
Utilies for general purposes
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com> Michele Ferretti <mic.ferretti@gmail.com>


import osmnx
from shapely import geometry
import geopandas as gpd
from geol.utils import constants
import json

def get_area_boundary(area_name, which_result=1):

    boundary = osmnx.gdf_from_place(area_name)

    if isinstance(boundary.loc[0]['geometry'], geometry.Point):

        boundary = osmnx.gdf_from_place(area_name, which_result=2)
        """
        boundary.loc[:, "isBBox"] = False

        # Build GeoDataFrame with the bounding box of the city
        topLeft = Point(boundary['bbox_west'].item(),
                        boundary['bbox_north'].item())
        bottomLeft = Point(boundary['bbox_west'].item(), boundary[
                           'bbox_south'].item())
        topRight = Point(boundary['bbox_east'].item(),
                         boundary['bbox_north'].item())
        bottomRight = Point(boundary['bbox_east'].item(), boundary[
                            'bbox_south'].item())

        pList = [topLeft, topRight, bottomRight, bottomLeft]

        boundary.loc[0]['geometry'] = [Polygon([[p.x, p.y] for p in pList])]
        boundary.loc[0]['isBBox'] = True
        """

    return boundary


def build_bbox(area, bbox_side_len=500):
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
    bbox = geometry.box(SW[0], SW[1], NE[0], NE[1], ccw=True)
    poly_df = gpd.GeoDataFrame(geometry=[bbox])
    poly_df.crs = {'init': 'epsg:' + constants.universal_crs, 'units': 'm'}

    return poly_df

def read_foursqaure_keys(filename):

    with open(filename) as json_data_file:
        data = json.load(json_data_file)

    return data