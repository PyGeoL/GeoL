"""
Utilies for general purposes
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com> Michele Ferretti <mic.ferretti@gmail.com>


import osmnx
from shapely.geometry import Point, Polygon


def get_area_boundary(area_name, which_result=1):

    boundary = osmnx.gdf_from_place(area_name)

    if isinstance(boundary.loc[0]['geometry'], Point):

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
