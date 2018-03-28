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

    # load Features DataFrame
    features_df = pd.read_csv(
        features_path, sep='\t', header=None)
    cols = [int(i) for i in features_df.columns]
    cols[0] = 'cellID'
    features_df.columns = cols
    features_df.columns = list(map(
        lambda x: 'f_fs_' + str(x) if x != "cellID" else x, features_df.columns))
    features_df.head(2)

    # load Targets DataFrame
    targets_df = pd.read_csv(targets_path)
    targets_df.columns = list(
        map(lambda x: 't_'+x if x != "cellID" else x, targets_df.columns))

    # select only relevant columns
    targets_df_relevant = targets_df.loc[:, [
        'cellID', 't_predominant']]

    # Merge Features and Targets
    if merge_strategy not in [1, 2, 3]:
        logger.info(
            "Please select a correct merge strategy. Options are (1) Left, (2) Right, (3) Inner Join.")
    if merge_strategy == 1:
        # Merge the UA and cellvector dataframes
        merged_features_targets = targets_df_relevant.merge(
            features_df, on="cellID", how='left')

    elif merge_strategy == 2:
        # Merge the UA and cellvector dataframes
        merged_features_targets = targets_df_relevant.merge(
            features_df, on="cellID", how='left')

    else:
        # Merge the UA and cellvector dataframes
        merged_features_targets = targets_df_relevant.merge(
            features_df, on="cellID", how='left')

    # remove empty
    merged_features_targets.dropna(inplace=True)

    return merged_features_targets


def split_train_test(merged_features_targets, features_path, targets_path, output_dir):

    # Divide train/test General
    df_feat = merged_features_targets[[x for x in merged_features_targets.columns if x.startswith(
        'f_')]+['cellID']].set_index('cellID')
    df_target = merged_features_targets[[x for x in merged_features_targets.columns if x.startswith(
        't_')]+['cellID']].set_index('cellID')

    print("==================", merged_features_targets.shape,
          df_feat.shape, df_target.shape)

    # Split Train and Test
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
        df_feat, df_target, test_size=0.2, random_state=42, stratify=df_target)

    # Remove empty values
    df_X_train.dropna(inplace=True)
    df_X_test.dropna(inplace=True)

    # Prepare output directory path
    OUTPUT_PATH = os.path.abspath(output_dir)
    OUTPUT_TRAIN = os.path.join(
        OUTPUT_PATH, "train_" + features_path.split("/")[-1].split(".")[0])
    OUTPUT_TEST = os.path.join(
        OUTPUT_PATH, "test_" + targets_path.split("/")[-1].split(".")[0])

    # Save datasets
    df_train = df_X_train.merge(df_y_train, left_index=True, right_index=True)
    df_test = df_X_test.merge(df_y_test, left_index=True, right_index=True)
    df_train.to_csv(OUTPUT_TRAIN, index_label="cellID",
                    sep="\t", float_format='%.6f')
    df_test.to_csv(OUTPUT_TEST, index_label="cellID",
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

    return df_train, df_test


# -----------------------------------------------------------------------------------------------------------


def main(argv):

    parser = argparse.ArgumentParser('Run XGBOOST on Cellvector embeddings')

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
                        help='Output directory. Train/Test files will be name with the following convention: test_/train_ + INPUT_FILE_NAME',
                        action='store',
                        dest='output_dir',
                        required=True,
                        type=str)

    parser.add_argument('-ms', '--merge-strategy',
                        help='Choose how to merge features and targets\' sets before splitting. 1 = left join, 2 = right, 3 = inner. Defaults to left joining features dataframe with targets.',
                        dest='merge_strategy',
                        default=1,
                        type=int)

    args = parser.parse_args()

    # Merge Features and Targets
    merged_features_targets = merge_features_targets(
        args.features_path, args.targets_path, merge_strategy)

    # Split and save the Train and Test datasets
    split_train_test(merge_features_targets, args.features_path,
                     args.targets_path, args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
