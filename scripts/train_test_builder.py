"""
Script to split a dataset in train+test sets.
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com> Michele Ferretti <mic.ferretti@gmail.com>


import argparse
import os
import math
import errno
import sys
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import getopt
import pathlib
import re
import gensim
import numpy as np
import logging
from geol.geol_logger.geol_logger import logger


def features_prepper(features_dir):
    """Prepares features in input

    :param features_dir: [Directory containing (multiple) feature file(s)]
    :type features_dir: [str]
    :return: [Dataframe of sanitised and concatenated features]
    :rtype: [Pandas Dataframe]
    """

    sanitized_features_dfs = []

    for features_df_name in os.listdir(features_dir):
        # load Features DataFrame + set cellID as index
        features_df = pd.read_csv(os.path.join(
            features_dir, features_df_name), sep="\t", index_col=["cellID"])
        # sanitizes column names
        sanitized_features_df = sanitize_features(
            features_df, features_df_name.split(".")[0])
        # store sanitized dfs
        sanitized_features_dfs.append(sanitized_features_df)

     # concat dataframes
    concat_sanitized_features_dfs = pd.concat(
        sanitized_features_dfs, axis=1)

    # remove index name
    del concat_sanitized_features_dfs.index.name
    return concat_sanitized_features_dfs


def train_test_loader(train_path, test_path):
    """load train and te    
    print(concat_sa)st files
    print(train_df)st files
    print(test_df)st files


    :param train_path: [Path to train file], defaults to args.train_path
    :param train_path: [str], 
    :param test_path: [Path to test file], defaults to args.test_path
    :param test_path: [str], 
    """

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")
    return train_df, test_df


def sanitize_features(df, prepend_letter):
    """Sanitizes column names and prepends custom strings to column names

    :param df: Pandas DataFrame to sanitize
    :type df: [Pandas DataFrame]
    :param prepend_letter: String to prepend
    :type prepend_letter: [str]
    """

    col_names = df.columns
    new_col_names = [prepend_letter+"_" + name.lower().replace(" ", "")
                     for name in col_names]
    df.columns = new_col_names
    return df


# -----------------------------------------------------------------------------------------------------------


def main(argv):

    parser = argparse.ArgumentParser('s')

    parser.add_argument('-f', '--features-dir',
                        help='Path to directory containing features files',
                        action='store',
                        dest='features_dir',
                        required=True,
                        nargs="?",
                        type=str)

    parser.add_argument('-tr', '--train-path',
                        help='Path to train/test ID files.',
                        action='store',
                        dest='train_path',
                        type=str)

    parser.add_argument('-te', '--test-path',
                        help='Path to test ID files. ',
                        action='store',
                        dest='test_path',
                        type=str)

    parser.add_argument('-o', '--output_dir',
                        help='Output directory. Train/Test files will be named with the following convention: test_/train_ + INPUT_FILE_NAME',
                        action='store',
                        dest='output_dir',
                        required=True,
                        type=str)

    parser.add_argument('-c', '--city-name',
                        help='City Name',
                        action='store',
                        dest='city_name',
                        required=True,
                        type=str)

    parser.add_argument('-s', '--size',
                        help='Grid cell Size',
                        action='store',
                        dest='grid_size',
                        required=True,
                        type=int)

    args = parser.parse_args()

    # NOTE: per adesso mancano i cell vector ed i count da usare come imput e concaternarli.
    # una volta che gianni li ha fatti puoi testare.
    # per ora uso dei dataframe finti.

    # load, sanitize and concat features files
    all_features = features_prepper(args.features_dir)

    # load train and test files
    train_IDs, test_IDs = train_test_loader(args.train_path, args.test_path)

    # join train/test with features
    #  Filling empty rows with 0
    train_features = train_IDs.merge(
        all_features, left_on="cellID", right_index=True, how='inner').reset_index(drop=True).fillna(0)

    test_features = test_IDs.merge(
        all_features, left_on="cellID", right_index=True, how='inner').reset_index(drop=True).fillna(0)
    print(test_features.head())
    # save train/test files
    OUTPUT_PATH_TRAIN = os.path.join(
        args.output_dir, args.city_name + "_train_s"+str(args.grid_size) + ".csv")
    OUTPUT_PATH_TEST = os.path.join(
        args.output_dir, args.city_name + "_test_s"+str(args.grid_size) + ".csv")
    train_features.to_csv(OUTPUT_PATH_TRAIN, index=False,
                          sep="\t", float_format='%.6f')
    test_features.to_csv(OUTPUT_PATH_TEST, index=False,
                         sep="\t", float_format='%.6f')


if __name__ == "__main__":
    main(sys.argv[1:])
