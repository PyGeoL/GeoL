# coding: utf-8

# map dataset entries onto the correct cell
# EPSG:4326  XY projected coords
# EPSG:3857 lat,lon spheric coords

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os
import getopt
import sys
import csv


def main(argv):

    try:
        opts, args = getopt.getopt(
            argv, "hg:d:o:n:", ["grid=", "dataset=", "outputfile=", "names="])

    except getopt.GetoptError:
        print(
            'script.py -g <grid> -d <dataset> -o <outputfile> -n <Latitude and Longitude names>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(
                'script.py -g <grid> -d <dataset> -o <outputfile> -n <Latitude and Longitude names>')
            sys.exit()
        elif opt in ("-g", "--grid"):
            grid_inputfile = os.path.abspath(arg)
        elif opt in ("-d", "--dataset"):
            dataset_inputfile = os.path.abspath(arg)
        elif opt in ("-o", "--outputfile"):
            outputfile = os.path.abspath(arg)
        elif opt in ("-n", "--names"):
            names = arg

    latitude, longitude = names.split(' ')[:2]

    # CARICO GRIGLIA E SETTO SISTEMA DI RIFERIMENTO A WGS84
    gdf = gpd.GeoDataFrame.from_file(grid_inputfile)
    # gdf.crs = {'init': 'epsg:2236', 'units': 'm'}
    gdf.crs = {'init': 'epsg:4326'}
    gdf = gdf.to_crs({'init': 'epsg:4326'})
#     print("STAMPO HEAD DI {}".format(grid_inputfile))
#     print(gdf.head(2))
    print('griglia caricata')

    # CARICO DATASET
    df = pd.DataFrame(pd.read_csv(
        dataset_inputfile, sep=",", low_memory=False))
    print('dataset_inputfile', dataset_inputfile)
    print(df.columns)

    print('dataset caricato')

    # CREO OGGETTI POINT CON LE COPPIE DI LONGITUDINE E LATITUDINE
    geometry = [Point(xy) for xy in zip(df[longitude], df[latitude])]

    # METTO I PUNTI IN UN GEODATAFRAME
    data = gpd.GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=geometry)
    # data = data.to_crs({'init': 'epsg:4326'})
    data.to_crs(gdf.crs, inplace=True)

    #  ALTERNATIVO: Ho di gia' delle features
    # data = gpd.GeoDataFrame.from_file(dataset_inputfile)
#     print(data.head())
    print('geodataframe costruito')

    # CHECK GEOMETRY VALIDITY

    ans = data.geometry.is_valid
    invalid = ans[ans == False]
    data.drop(invalid.index, axis=0, inplace=True)
    print(data.head())

    print('punti non validi filtrati')

    # FACCIO UNA SPACIAL JOIN CON LA GRIGLIA PER ASSOCIARE OGNI ENTRY ALLA
    # CELLA CHE LA CONTIENE ('within')
    join = gpd.sjoin(data, gdf[['cellID', 'geometry']],
                     how='inner', op='within')

    print(join.head())
    print('spatial join terminata')

    # ELIMINO LE COLONNE NON PIÙ NECESSARIE
    join.drop(['index_right', 'geometry'], axis=1, inplace=True)

    # E RINOMINO L' 'id' DELLE CELLE
    # join.rename(columns={'id': 'cellID'}, inplace=True)
    print(join.head())

    # INFINE, SCRIVO IL FILE
    join.to_csv(outputfile, index=False, sep='\t', float_format='%.6f')

    print('file salvato')


if __name__ == "__main__":
    main(sys.argv[1:])
