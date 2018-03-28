# coding: utf-8

# map dataset entries onto the correct cell
# EPSG:4326  XY projected coords
# EPSG:3857 lat,lon spheric coords

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geol.geol_logger.geol_logger import logger
import sys
import argparse
import os


def main(argv):

    parser = argparse.ArgumentParser('Foursquare crawler.')

    parser.add_argument('-i', '--input',
                        help='POIs file with relative coordinates.',
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

    parser.add_argument('-lat', '--latitude',
                        help='Latitude name.',
                        action='store',
                        dest='latitude',
                        default='latitude',
                        type=str)

    parser.add_argument('-long', '--longitude',
                        help='Longitude name.',
                        action='store',
                        dest='longitude',
                        default='longitude',
                        type=str)

    args = parser.parse_args()

    latitude = args.latitude
    longitude = args.longitude

    if(args.verbosity == 1):
        logger.setLevel()

    elif(args.verbosity == 2):
        logger.setLevel(logger.INFO)


    # Load the grid
    logger.info("Load the grid")
    gdf = gpd.GeoDataFrame.from_file(args.grid)
    gdf.crs = {'init': args.crs}

    if args.crs != 'epsg:4326':
        gdf = gdf.to_crs({'init': 'epsg:4326'})

    # Load POIs
    logger.info("Load POIs")
    df = pd.DataFrame(pd.read_csv(args.input, sep=",", low_memory=False))

    # Create Point from latitude, longitude pairs and build a GeoDataFrame
    logger.info("Build geometry")
    geometry = [Point(xy) for xy in zip(df[longitude], df[latitude])]
    data = gpd.GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=geometry)
    data.to_crs(gdf.crs, inplace=True)

    # Check Geometry Validity
    ans = data.geometry.is_valid
    invalid = ans[ans == False]
    data.drop(invalid.index, axis=0, inplace=True)

    # Spatial Join with the grid to associate each entry to the related cell ('within')
    join = gpd.sjoin(data, gdf[['cellID', 'geometry']], how='inner', op='within')

    # Remove additional columns
    join.drop(['index_right', 'geometry'], axis=1, inplace=True)

    # Save output
    logger.info("Save output file")
    outputfile = os.path.abspath(os.path.join(args.outputfolder, args.prefix + "_mapped_foursquare_pois.csv"))
    join.to_csv(outputfile, index=False, sep='\t', float_format='%.6f')

if __name__ == "__main__":
    main(sys.argv[1:])
