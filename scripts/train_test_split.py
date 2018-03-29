"""
Script to split a dataset in train+test sets.
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com> Michele Ferretti <mic.ferretti@gmail.com>

import argparse
import os
import math
import errno
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import sys
sys.path.append("../GeoL")
import getopt

import pathlib
import re
import gensim

import numpy as np
from sklearn import preprocessing

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


from scipy import stats

import seaborn as sns
sns.set_style("ticks")
sns.set_context("paper")


import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import sklearn
from sklearn.model_selection import train_test_split
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

# import logging
# logger = logging.getLogger(__name__)
from geol.geol_logger.geol_logger import logger

# def sanitize_features(df, prepend_letter):
#     """Sanitizes column names and prepends custom strings to column names

#     :param df: Pandas DataFrame to sanitize
#     :type df: [Pandas DataFrame]
#     :param prepend_letter: String to prepend
#     :type prepend_letter: [str]
#     """

#     col_names = df.columns
#     new_col_names = [prepend_letter+"_" + name.lower().replace(" ", "")
#                      for name in col_names]
#     df.columns = new_col_names


def merge_features_targets(features_path, targets_path, merge_strategy):
    """
    Reads Features and Targets Dataframes (Urban Atlas data)
    Merges them according to the provided merge strategy
    """

    # load Targets DataFrame
    targets_df = pd.read_csv(
        targets_path)

    # load Features DataFrame
    features_df = pd.read_csv(features_path, sep="\t")

    # select only relevant columns
    targets_df_relevant = targets_df.loc[:, ['cellID', 'predominant']]
    features_df_relevant = features_df.loc[:, ['cellID']]
    features_df_relevant.drop_duplicates(
        subset=["cellID"], inplace=True)
    # Merge Features and Targets
    if merge_strategy not in [0, 1]:
        logger.info(
            "Please select a correct merge strategy. Options are (1) Left, (2) Right, (3) Inner Join.")
    if merge_strategy == 0:
        # Merge the UA and cellvector dataframes
        merged_df = targets_df_relevant.merge(
            features_df_relevant, on="cellID", how='left')
    else:
        # Merge the UA and cellvector dataframes
        merged_df = targets_df_relevant.merge(
            features_df_relevant, on="cellID", how='inner')
    return merged_df


def split_train_test(merged_df, features_path, targets_path, output_dir):

    # Divide train/test
    # NOTE: I'm repeating the IDs as Sklearn wants 2 separate inputs
    df_feat = merged_df['cellID'].values
    df_target = merged_df['predominant'].values

    # Split Train and Test
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
        df_feat, df_target, test_size=0.2, random_state=42, stratify=df_target)
    print(type(df_X_train))

    # # Remove empty values
    # df_X_train.dropna(inplace=True)
    # df_X_test.dropna(inplace=True)

    # Prepare output directory path
    OUTPUT_PATH = os.path.abspath(output_dir)
    OUTPUT_TRAIN = os.path.join(
        OUTPUT_PATH, "train_IDs_" + features_path.split("/")[-1].split(".")[0] + ".csv")
    OUTPUT_TEST = os.path.join(
        OUTPUT_PATH, "test_IDs_" + targets_path.split("/")[-1].split(".")[0] + ".csv")

    # Save datasets
    df_train = pd.DataFrame({"cellID": df_X_train, "predominant": df_y_train})
    df_test = pd.DataFrame({"cellID": df_X_test, "predominant": df_y_test})
    # df_train = df_X_train.merge(
    # df_y_train, left_index=True, right_index=True)
    # df_test = df_X_test.merge(df_y_test, left_index=True, right_index=True)
    df_train.to_csv(OUTPUT_TRAIN, index=False,
                    sep="\t", float_format='%.6f')
    df_test.to_csv(OUTPUT_TEST, index=False,
                   sep="\t", float_format='%.6f')

    # # Create scaled version of train and test
    # X_scaler = StandardScaler()
    # df_X_train_scaled = X_scaler.fit_transform(df_X_train)
    # df_X_train_scaled = pd.DataFrame(
    #     df_X_train_scaled, index=df_X_train.index, columns=df_X_train.columns)
    # df_X_test_scaled = X_scaler.transform(df_X_test)
    # df_X_test_scaled = pd.DataFrame(
    #     df_X_test_scaled, index=df_X_test.index, columns=df_X_test.columns)

    # df_train_scaled = df_X_train_scaled.merge(
    #     df_y_train, left_index=True, right_index=True)
    # df_test_scaled = df_X_test_scaled.merge(
    #     df_y_test, left_index=True, right_index=True)

    # df_train_scaled.to_csv(OUTPUT_TRAIN_SCALED,
    #                        index_label="cellID", sep="\t", float_format='%.6f')
    # df_test_scaled.to_csv(OUTPUT_TEST_SCALED,
    #                       index_label="cellID", sep="\t", float_format='%.6f')

    # return df_train, df_test


# -----------------------------------------------------------------------------------------------------------


def main(argv):

    parser = argparse.ArgumentParser('s')

    parser.add_argument('-f', '--features-path',
                        help='Path to features file',
                        action='store',
                        dest='features_path',
                        required=True,
                        type=str)

    parser.add_argument('-t', '--targets-path',
                        help='Path to targets file',
                        action='store',
                        dest='targets_path',
                        required=True,
                        type=str)

    parser.add_argument('-o', '--output_dir',
                        help='Output directory. Train/Test files will be name with the following convention: test_IDs_/train_IDs_ + INPUT_FILE_NAME',
                        action='store',
                        dest='output_dir',
                        required=True,
                        type=str)

    parser.add_argument('-ms', '--merge-strategy',
                        help='Choose how to merge features and targets\' sets before splitting. 0 = left join, 1 = inner. Defaults to inner, joining features dataframe with targets.',
                        dest='merge_strategy',
                        default=1,
                        type=int)

    args = parser.parse_args()

    # Merge Features and Targets
    merged_df = merge_features_targets(
        args.features_path, args.targets_path, args.merge_strategy)

    # Split and save the Train and Test datasets
    split_train_test(merged_df, args.features_path,
                     args.targets_path, args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])

# TODO
# devi testartlo ma per farlo hai bisogno dei dati di FS mapped sulla griglia che stai rifacendo perche gianni ha cancellato tutto
# poi potrai lanciar il ttto con :
# python scripts/train_test_split.py - f - t data/barcelona/count/barcelona_ua_count_200.csv - o test
