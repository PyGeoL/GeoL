
# coding: utf-8
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn import preprocessing
pd.options.display.max_colwidth = 1000
import sys
from geol.geol_logger.geol_logger import logger
import csv
import argparse
from geol.utils import constants

def main(argv):

    parser = argparse.ArgumentParser('Grid - UrbanAtlasLandUse mapping.')

    parser.add_argument('-i', '--input',
                        help='Landuse shape files.',
                        action='store',
                        dest='input',
                        required=True,
                        type=str)

    parser.add_argument('-g', '--grid',
                        help='Input grid for the mapping. If crs is not WGS84, specify it with the param -c',
                        action='store',
                        dest='grid',
                        required=True,
                        type=str)

    parser.add_argument('-c', '--crs',
                        help='Coordinate Reference System for the input grid. It is requested only if it is different from WGS84.',
                        action='store',
                        dest='crs',
                        default='epsg:4326',
                        type=str)

    parser.add_argument('-o', '--outputfolder',
                        help='Output folder where to save the mapped file.',
                        action='store',
                        dest='outputfolder',
                        required='True',
                        type=str)

    args = parser.parse_args()

    if(args.verbosity == 1):
        logger.setLevel()

    elif(args.verbosity == 2):
        logger.setLevel(logger.INFO)

    # Read UA dataset
    landuse = gpd.GeoDataFrame.from_file(args.input)

    # fill NaN
    landuse["ITEM2012"] = landuse["ITEM2012"].fillna('Undefined')

    # ----------- Collapsing landuse classes -----------
    # HD
    landuse["ITEM2012"].replace(to_replace="Continuous urban fabric (S.L. : > 80%)",
                                value="HD", inplace=True)
    landuse["ITEM2012"].replace(to_replace="Discontinuous dense urban fabric (S.L. : 50% -  80%)",
                                value="HD", inplace=True)
    # MD
    landuse["ITEM2012"].replace(to_replace="Discontinuous medium density urban fabric (S.L. : 30% - 50%)",
                                value="MD", inplace=True)

    # LD
    landuse["ITEM2012"].replace(to_replace="Discontinuous low density urban fabric (S.L. : 10% - 30%)",
                                value="LD", inplace=True)
    landuse["ITEM2012"].replace(to_replace="Discontinuous very low density urban fabric (S.L. : < 10%)",
                                value="LD", inplace=True)

    # Agri
    landuse["ITEM2012"].replace(to_replace="Arable land (annual crops)",
                                value="Agri", inplace=True)
    landuse["ITEM2012"].replace(to_replace="Permanent crops (vineyards, fruit trees, olive groves)",
                                value="Agri", inplace=True)
    landuse["ITEM2012"].replace(
        to_replace="Pastures", value="Agri", inplace=True)
    landuse["ITEM2012"].replace(
        to_replace="Complex and mixed cultivation patterns", value="Agri", inplace=True)
    landuse["ITEM2012"].replace(
        to_replace="Orchads", value="Agri", inplace=True)
    landuse["ITEM2012"].replace(
        to_replace="Wetlands", value="Agri", inplace=True)
    landuse["ITEM2012"].replace(to_replace="Open spaces with little or no vegetation (beaches, dunes, bare rocks, glaciers)",
                                value="Agri", inplace=True)
    landuse["ITEM2012"].replace(to_replace="Herbaceous vegetation associations (natural grassland, moors...)",
                                value="Agri", inplace=True)
    # SPORTS
    landuse["ITEM2012"].replace(to_replace="Sports and leisure facilities",
                                value="Sports", inplace=True)
    # INDUSTRIAL
    landuse["ITEM2012"].replace(to_replace="Industrial, commercial, public, military and private units",
                                value="Industrial", inplace=True)

    # Green Urban
    landuse["ITEM2012"].replace(to_replace="Green urban areas",
                                value="Green_Urban", inplace=True)

    # Transport
    landuse["ITEM2012"].replace(to_replace="Fast transit roads and associated land",
                                value="Transport", inplace=True)
    landuse["ITEM2012"].replace(to_replace="Other roads and associated land",
                                value="Transport", inplace=True)
    landuse["ITEM2012"].replace(to_replace="Railways and associated land",
                                value="Transport", inplace=True)
    landuse["ITEM2012"].replace(to_replace="Port areas",
                                value="Transport", inplace=True)
    landuse["ITEM2012"].replace(to_replace="Airports",
                                value="Transport", inplace=True)

    # Subset of land use classes (gonalez's classes)
    admitted_classes = ['Sports', 'HD', 'MD', 'LD', 'Industrial', 'Green_Urban', 'Forests', 'Transport', 'Agri']

    # ----------- End collapsing landuse classes -----------

    # We are removing the NOT admitted classes after computing the predominant
    # landuse_admitted = landuse[landuse['ITEM2012'].isin(admitted_classes)]

    landuse_admitted = landuse.to_crs({'init': constants.universal_crs})[['ITEM2012', 'geometry']]

    # drona
    landuse_admitted.dropna(inplace=True)

    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()
    reshaped_geom = np.array(landuse_admitted.geometry.area / 10**6).reshape(-1, 1)
    x_scaled = min_max_scaler.fit_transform(reshaped_geom)
    landuse_admitted.loc[:, 'coverage'] = x_scaled(landuse_admitted.groupby(['ITEM2012']).sum() /
                                                    landuse_admitted['coverage'].sum())

    # Load Empty Grid
    grid = gpd.GeoDataFrame.from_file(args.grid)
    grid = grid.to_crs({"init": constants.universal_crs})

    landuse_admitted.geometry = landuse_admitted.geometry.buffer(0)

    landuse_rows_initial = landuse_admitted.shape[0]
    grid_rows_initial = grid.shape[0]

    # Spatial join
    df = gpd.sjoin(grid, landuse_admitted, op='intersects')
    df = df.merge(landuse_admitted[['geometry']], left_on='index_right', right_index=True)
    df.loc[:, 'intersection'] = df.apply(lambda row: row['geometry_y'].intersection(row['geometry_x']), axis=1)

    #landuse_rows_sjoin = landuse_admitted.shape[0]
    #grid_rows_sjoin = grid.shape[0]

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
    # Note: it repeates values for columns area_tot, since temp DataFrame has < rows than df DataFrame

    df = df.merge(temp, on='cellID')
    df['percentage'] = df['area'] / grid.loc[0].geometry.area
    df['normalized_percentage'] = df['area'] / df['area_tot']

    landuse_rows_merge = landuse_admitted.shape[0]
    grid_rows_merge = grid.shape[0]

    # Since we care only about general landuse type for each cell, we group by cellID and LandUse,
    # summing the percentage of each activity with the same type in each cell
    df = df[['normalized_percentage', 'percentage', 'cellID',
             'ITEM2012']].groupby(['cellID', 'ITEM2012']).sum()
    df = df.reset_index()

    # Compute the list of landuse column names
    lu_col = df['ITEM2012'].drop_duplicates()

    r = pd.pivot_table(df, values='percentage', index=['cellID'], columns=['ITEM2012']).reset_index()
    r.fillna(0, inplace=True)
    r.loc[:, "predominant"] = r[lu_col].idxmax(axis=1)

    # Filter out cell where predominant is more than 0.25 of total area in the cell
    r.loc[:, "valid"] = r.apply(lambda x: 1 if x[x['predominant']] > 0.25 else 0, axis=1)

    # Take only valid columns
    r_valid = r[r["valid"] != 0]

    # Select only cell in admitted classes - IMPORTANT PASSAGE: we are removing the NOT admitted classes
    # after computing the predominant for each cell.
    r_valid = r_valid[r_valid['predominant'].isin(admitted_classes)]

    # Write mapped output
    outputfile = os.path.abspath(os.path.join(args.outputfolder, args.prefix + "_mapped_ua.csv"))

    r_valid.to_csv(outputfile, index=False, quoting=csv.QUOTE_NONNUMERIC)
    check_statistics = """
        landuse_rows_initial:  {}
        grid_rows_initial: {}
        landuse_rows_merge: {}
        grid_rows_merge: {}
    """.format(landuse_rows_initial, grid_rows_initial, landuse_rows_merge, grid_rows_merge)

    with open(outputfile, 'w') as outputfile_stats:
        outputfile_stats.write(check_statistics)

if __name__ == "__main__":
    main(sys.argv[1:])
