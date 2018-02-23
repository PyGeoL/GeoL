"""
Methods to load different landuse datasets.
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

import geopandas as gpd
import os
from ..utils import constants


def urban_atlas(input, aggregation_level=0):

    gdf = gpd.read_file(os.path.abspath(input))
    gdf[constants.ITEM_COL] = gdf[constants.ITEM_COL].fillna('Undefined')

    if aggregation_level == 1:
        for k,v in constants.urbanatlas_aggregation:
            for e in v:
                gdf[constants.ITEM_COL].replace(to_replace=k, value=e, inplace=True)

    gdf = gdf.to_crs({'init': 'epsg:4326'})[[constants.CITIES, constants.ITEM_COL, 'geometry']]

    return gdf






    def compute_landuse(self, landuse):

        df = gpd.sjoin(self.grid, landuse, how='left', op='intersects')
        df = df.merge(landuse[['geometry']], left_on='index_right', right_index=True)
        df.loc[:, 'intersection'] = df.apply(lambda row: row['geometry_y'].intersection(row['geometry_x']), axis=1)

        df = df.reset_index()[['cellID', 'ITEM2012', 'intersection']]
        df.rename(columns={'intersection': 'geometry'}, inplace=True)
        df = df.set_geometry('geometry')

        # Compute the column with areas of each landuse in each cell
        df['area'] = df.geometry.area

        # In order to compute the total area in each cell, group by cellID and sum the area in each group
        temp = df[['cellID', 'area']].groupby('cellID').sum()

        temp.rename(columns={'area': 'area_tot'}, inplace=True)
        temp = temp.reset_index()

        # Merge the temporary dataframe just created with the original one
        df = df.merge(temp, on='cellID')
        df['percentage'] = df['area'] / self.grid.loc[0].geometry.area
        df['normalized_percentage'] = df['area'] / df['area_tot']

        # Since we care only about general landuse type for each cell, we group by cellID and LandUse,
        # summing the percentage of each activity with the same type in each cell
        df = df[['normalized_percentage', 'percentage', 'cellID', constants.ITEM_COL]].groupby(['cellID', constants.ITEM_COL]).sum()
        df = df.reset_index()

        # Compute the list of landuse column names
        lu_cols = df[constants.ITEM_COL].drop_duplicates()

        r = pd.pivot_table(df, values='percentage', index=['cellID'], columns=[constants.ITEM_COL]).reset_index()
        r.fillna(0, inplace=True)
        r.loc[:, "predominant"] = r[lu_cols].idxmax(axis=1)

        lu_cols.append("predominant")
        lu_cols.append("percentage")
        lu_cols.append("normalized_percentage")

        self.grid.merge(r, on='cellID')

        self.columns['landuse'] = lu_cols
