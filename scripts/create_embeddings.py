import os
import errno
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import sys
import getopt
sys.path.append('./GeoL')
from geol.utils import utils
import pathlib
import re
import gensim

import numpy as np
from sklearn import preprocessing

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats

import seaborn as sns
sns.set_style("ticks")
sns.set_context("paper")

from sklearn.model_selection import train_test_split
import xgboost
from xgboost.sklearn import XGBRegressor
from sklearn import metrics  # Additional scklearn functions
from sklearn.model_selection import GridSearchCV  # Perforing grid search


from sklearn.svm import SVR
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 4

import csv


# SET CITY NAME
# ,"milan","madrid","berlin","paris","stockholm"]
CITIES = ["barcelona", "london", "rome"]


SIZES = [50, 100, 200, 250]


# Base directory
BASE_DIR = os.path.abspath(".")
# base directory for data files
BASE_DIR_DATA = os.path.join(BASE_DIR, "data")


def csv_2_geodf(dataset_inputfile):
    """
    Loads a .csv and returns a GeoPandas GeoDataFrame
    """
    # LOAD DATASET
    df = pd.DataFrame(pd.read_csv(
        dataset_inputfile, sep="\t", low_memory=False))
    print('dataset_inputfile', dataset_inputfile)
    print(df.columns)

    print('dataset caricato')

    # Create Shapely Point Objects
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]

    # Store Points in a GeoDataFrame
    data = gpd.GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=geometry)

    # data = data.to_crs({'init': 'epsg:4326'})
    data.to_crs(data.crs, inplace=True)
    return data


def sum_vectors(w_list):
    """
    Inputs a list of Numpy vectors
    Returns the sum 
    """
    e = 0
    for i, e in enumerate(w_list):
        e += w_list[i]
    return e


def cell_vector_representation(poi_grid, w2v_model, level, output_file, size):
    """
    Takes as input a spatial grid with POIs for each cell, a Word2Vec model, and a level of detail
    For each cell:
        Looks up each category in a cell for the given level in the W2V model, taking the corresponding vector representation
        Sums all the vectors
    Returns a dataframe with a w2v representation for all words in that cell in every row
    """
    # load shapefile of mapped POIs
    gdf = csv_2_geodf(poi_grid)

    # load w2v_model
    model = gensim.models.Word2Vec.load(w2v_model)

    # group every cell
    grouped_gdf = gdf.groupby('cellID')

    output = {}
    with open(output_file, 'w') as out:
        for cell, group in grouped_gdf:
            output[cell] = []
            for categories_raw in group['categories']:
                # select level
                category = utils.select_category(
                    categories_raw.split(':'), level)[-1]
                # lookup category in w2v
                try:
                    vector = model[category]
                    output[cell].append(np.array(vector))
                except(KeyError):
                    pass
            if len(output[cell]) == 0:
                output[cell] = [np.zeros(int(size))]

            # sum vectors
            sum_w = sum_vectors(output[cell])
            sum_w_str = str("\t".join(map(str, sum_w)))
            text_to_write = str(cell) + '\t' + sum_w_str + '\n'

            out.write(text_to_write)


for CITY in CITIES:

    BASE_DIR_CITY = os.path.join(BASE_DIR_DATA, CITY)
    MODELS_DIR = os.path.join(BASE_DIR_CITY, 'output-skip', 'models')
    GRID_DIR = os.path.join(BASE_DIR_CITY, 'mapped')
    EMBEDDINGS_DIR = os.path.join(BASE_DIR_CITY, 'embeddings')

    for MAPPED_GRID in os.listdir(GRID_DIR):
        POI_GRID = os.path.join(GRID_DIR, MAPPED_GRID)
        POI_GRID_SIZE = MAPPED_GRID.split('.')[0].split('_')[-1]
        for MODEL in os.listdir(MODELS_DIR):
            W2V_MODEL = os.path.join(MODELS_DIR, MODEL)
            OUTPUT_NAME = CITY + "_gs" + POI_GRID_SIZE + \
                "_"+MODEL.split('.')[0] + '.txt'
            OUTPUT_PATH = os.path.join(EMBEDDINGS_DIR, OUTPUT_NAME)
            print(OUTPUT_PATH)
            # m = re.search('_s([0-9]+)_', MODEL)
            # if m:
            #     size = m.group(1)
            # print(OUTPUT_PATH)
            # cell_vector_representation(
            #     POI_GRID, W2V_MODEL, 2, OUTPUT_PATH, size)
