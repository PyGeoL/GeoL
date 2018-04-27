"""
Utilies for general purposes
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com> Michele Ferretti <mic.ferretti@gmail.com>


import osmnx
from shapely import geometry
import geopandas as gpd
from geol.utils import constants
import json
import os
import errno
import re


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
    # Make sure we are working with projected data
    area = area.to_crs(
        {'init':  constants.universal_crs, 'units': 'm'})
    # get area centroid
    centroid = area.centroid[0].coords[0]
    print(centroid)

    # get North-East corner
    NE = [float(coord)+(bbox_side_len/2) for coord in centroid]
    # get South-West corner
    SW = [float(coord)-(bbox_side_len/2) for coord in centroid]

    # build bbox from NE,SW corners
    bbox = geometry.box(SW[0], SW[1], NE[0], NE[1], ccw=True)
    print(bbox)
    poly_df = gpd.GeoDataFrame(geometry=[bbox])
    poly_df.crs = {'init': constants.universal_crs, 'units': 'm'}

    return poly_df


def assign_crs(grid, crs):
    print(crs)
    if crs is not None:
        return grid.to_crs({'init': crs})
    return grid


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def read_foursqaure_keys(filename):

    with open(filename) as json_data_file:
        data = json.load(json_data_file)

    return data


def normalize_word(word):

    pattern = re.compile('[\W]+', re.UNICODE)
    pattern.sub(r' ', word.lower()).strip().replace(" ", "_")

def normalize_words(words_array):
    """
    Remove nasty chars and sets word to lowercase
    """

    return [normalize_word(x) for x in words_array]


def select_category(list_of_labels, level):
    """
    Aggregates all the labels for each feature at the given level separated with _
    """
    tmp = []

    for x in list_of_labels:
        norm_w = normalize_words(x.split(":"))

        if len(norm_w) > level:
            tmp.append(norm_w[level])
        else:
            tmp.append(norm_w[len(norm_w) - 1])
    return tmp


def pre_processing(labels_list, depth_level=5):
    """
    Returns array of array of joined labels for specified depth level (default=5, last word of each chain.)
    """

    # select labels at given depth and join them
    labels_joined_list = [select_category(x, depth_level) for x in labels_list]

    return labels_joined_list
