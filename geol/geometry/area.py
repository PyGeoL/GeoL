"""
Class to deal with land use dataset provided by UrbanAtlas (https://land.copernicus.eu/local/urban-atlas/urban-atlas-2012/)
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com> Michele Ferretti <mic.ferretti@gmail.com>

# -----------------------------------------------------------------------------------------------
# TODO: Refactor/Remove
# -----------------------------------------------------------------------------------------------

# from geol.utils import constants, utils
# from ..factory import Square
# import geopandas as gpd
# from ..datasets import landuse as lu


# class Area:

#     def __init__(self, area_name, which_result,   input_shape=None):

#         if input_shape is not None:
#             self.area = gpd.read_file(input_shape)
#         else:
#             self.area = utils.get_area_boundary(area_name, which_result)
#         self.grid = None
#         self.name = area_name

#     def set_grid(self, input_grid=None, meters=None):

#         if input_grid is not None:
#             self.grid = Square(input=input_grid)
#         elif meters is not None:
#             self.grid = Square(base_shape=self.area)
#             self.grid.create_grid(meters=meters, inplace=True)
#         else:
#             raise ValueError(
#                 "If input_shape is None, then you must pass an %s value for meters." % type(int))

#     def add_landuse(self, input, source=None, aggregation_level=0, grid_map=False):

#         # TODO: manage general case when the user loads an unknown dataset.
#         if source == "UrbanAtlas":
#             self.landuse = lu.urban_atlas(input, aggregation_level)

#         if grid_map:
#             if self.grid is None:
#                 raise ValueError(
#                     "Mapping failed: the grid cannot be %s." % type(None))

#     def add_pois(self):
#         # Given an area and a list of POIs contained on it, add a dataframe with POIs attributes + cellID
#         # Load POIs and grid -> mapping -> save geodataframe
#         pass
