
# coding: utf-8


# Load FourSquare MAPPED dataset and assign to each square in the grid the
# number of POI of each FS category.

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt

import operator
import sys
import getopt


def category(df, level):
    tmp = df['categories'].split(":")

    if len(tmp) > level:
        return tmp[level]
    else:
        return tmp[len(tmp) - 1]


def main(argv):

    try:
        opts, args = getopt.getopt(
            argv, "hm:l:o:", ["map=", "level=", "outputfile"])

    except getopt.GetoptError:
        print ('script.py -m <map> -l <level> -o <outputfile>\n')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('script.py -m <map> -l <level> -o <outputfile>\n')
            sys.exit()
        elif opt in ("-m", "--map"):
            mapfile = arg
        elif opt in ("-l", "--level"):
            level_selection = int(arg)
        elif opt in ("-o", "--outputfile"):
            output = arg

    # load foursquare dataset mapped on a particular grid
    df = pd.read_csv(mapfile, sep='\t')
    df['categories'] = df['categories'].astype(str)

    # assign category to each record of the dataset
    df["categ"] = df.apply(category, args=(level_selection,), axis=1)
    # df.categories.apply(lambda x: x.split(":")[level_selection] if len(
    #     x.split(":")) > level_selection else None)

    # drop entry with empty category
    df = df.loc[df["categ"] != "nan"]
    categories = df["categ"].drop_duplicates().values

    # compute the list of FS categories
    POI = list(set(df['categ'].values))

    print('df len: ' + str(len(df)))

    # FOURSQUARE
    # compute aggregation on 'cellID', summing inside the columns 'checkin'
    # and 'usercount'
    # cell_df = df[['checkin', 'usercount', 'cellID']].groupby(['cellID']).sum()
    # OSM
    # compute aggregation on 'cellID', summing inside the columns 'checkin'
    # and 'usercount'
    cell_df = df.groupby(['cellID']).sum()

    cell_df.reset_index(inplace=True)

    # compute aggregation on 'cellID' and 'categ' to count the number of each
    # category in each cell
    cat_df = pd.DataFrame(df.groupby(
        ['cellID', 'categ']).size(), columns=['count'])
    cat_df.reset_index(inplace=True)
    #print('cat-df rows: ' + str(len(cat_df)))

    # now, create a table with the same information of 'cat_df' but using
    # categories names as columns names
    r = pd.pivot_table(cat_df, values='count', index=[
                       'cellID'], columns=['categ']).reset_index()

    # fill empty cells
    r[POI] = r[POI].fillna(0)

    # cast to int of every value
    r = r.astype(int)

    # merge the two table in order to get one table with all the informations
    df = cell_df.merge(r, on='cellID')

    # FOURSQUARE
    # sort columns
    # df = df[POI + ['cellID', 'checkin', 'usercount']]
    # OSM
    # sort columns
    df = df[POI + ['cellID']]

    # sort rows by cell id
    df.sort_values(by='cellID', inplace=True)

    # write to file
    df.to_csv(output, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
