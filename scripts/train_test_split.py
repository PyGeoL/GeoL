"""
Script to split a dataset in train+test IDs sets. 
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
from sklearn.model_selection import train_test_split
from geol.geometry.grid import Grid
from geol.geol_logger.geol_logger import logger


def filter_landuse(df, threshold=0.25):


    admitted_classes = ['Sports', 'HD', 'MD', 'LD', 'Industrial', 'Green_Urban', 'Transport']

    df_valid = df.copy()

    # Filter out cell where predominant is more than 0.25 of total area in the cell
    df_valid.loc[:, "valid"] = df_valid.apply(
        lambda x: 1 if x[x['predominant']] > 0.25 else 0, axis=1)

    # Take only valid columns
    df_valid = df_valid[df_valid["valid"] != 0]

    # Select only cell in admitted classes - IMPORTANT PASSAGE: we are removing the NOT admitted classes
    # after computing the predominant for each cell.
    df_valid = df_valid[df_valid['predominant'].isin(admitted_classes)]

    return df_valid[['cellID']]


def filter_pois(df, threshold=1):

    tmp = df[df.groupby('cellID').cellID.transform('size') > threshold]
    return tmp.drop_duplicates("cellID")[["cellID"]]


def merge_features_targets(features_path, targets_path, merge_strategy):
    """
    Reads Features and Targets Dataframes (Urban Atlas data)
    Merges them according to the provided merge strategy
    """

    # load Features DataFrame
    features_df = pd.read_csv(features_path, sep='\t', header=None)

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

    parser = argparse.ArgumentParser('s')

    parser.add_argument('-l', '--landuse',
                        help='Mapped land use filename',
                        action='store',
                        dest='landuse',
                        required=True,
                        type=str)

    parser.add_argument('-p', '--pois',
                        help='Mapped POIs filename',
                        action='store',
                        dest='pois',
                        required=True,
                        type=str)

    parser.add_argument('-g', '--grid',
                        help='Original grid filename',
                        action='store',
                        dest='grid',
                        required=True,
                        type=str)

    parser.add_argument('-f', '--features',
                        help='List of features file (f1, f2, ..)',
                        dest='features',
                        nargs="+",
                        type=str)

    parser.add_argument('-flu', '--filter_landuse',
                        help='Enable the removal of cells with too many land usese (< 25%)',
                        dest='flu',
                        action='store_true',
                        default=True)

    parser.add_argument('-fpoi', '--filter_pois',
                        help='Enable the removal of cells without POIs.',
                        dest='fpoi',
                        action='store_true',
                        default=True)

    parser.add_argument('-tlu', '--threshold_landuse',
                        help='Set the threshold for the most predominant class (0.25 by default).',
                        action='store',
                        dest='tlu',
                        default=0.25,
                        type=int)

    parser.add_argument('-tpoi', '--threshold_poi',
                        help='Set the threshold for the minimum number of POIs (default 1).',
                        action='store',
                        dest='tpoi',
                        default=1,
                        type=int)

    parser.add_argument('-o', '--output_dir',
                        help='Output directory. Train/Test files will be name with the following convention: test_/train_ + INPUT_FILE_NAME',
                        action='store',
                        dest='output_dir',
                        required=True,
                        type=str)

    parser.add_argument('-n', '--exp_name',
                        help='Name of the experiment. It will be used to rename train/text folders.',
                        action='store',
                        dest='exp_name',
                        required=True,
                        type=str)

    args = parser.parse_args()

    # Merge Features and Targets
    # merged_features_targets = merge_features_targets(args.features_path, args.targets_path, args.merge_strategy)

    # Load UA_mapped, POIs mapped and grid
    landuse_file = os.path.abspath(args.landuse)
    landuse = pd.read_csv(landuse_file)

    pois_file = os.path.abspath(args.pois)
    pois = pd.read_csv(pois_file, sep="\t")

    grid = Grid.from_file(args.grid)

    # Apply filters based on user input
    if args.flu == True:
        landuse_filtered = filter_landuse(landuse, args.tlu)

    if args.fpoi == True:
        pois_filtered = filter_pois(pois, args.tpoi)

    # Merge and create a list of final admitted cells
    admitted_cells = grid.grid.merge(landuse_filtered[["cellID"]], on="cellID", how="inner").merge(
        pois_filtered, on="cellID")[["cellID"]]

    # Load features file
    dfs = []
    for f in args.features:
        tmp = pd.read_csv(f, index_col="cellID")
        dfs.append(tmp)

    features = pd.concat(dfs, axis=1)

    # Keep only valid cells
    df_all = pd.concat(
        [features, landuse[["predominant", "cellID"]].set_index("cellID")], axis=1)
    df_all = df_all.filter(items=admitted_cells["cellID"], axis=0)

    # Split train/test taking into account the random seed to make reproducible the research
    df_feat = df_all[[x for x in df_all.columns if x.startswith('f_')]]
    df_target = df_all[["predominant"]].rename(
        columns={"predominant": "target"})

    # Split Train and Test
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
        df_feat, df_target, test_size=0.2, random_state=42, stratify=df_target)

    # Remove empty values
    # df_X_train.dropna(inplace=True)
    # df_X_test.dropna(inplace=True)

    # Prepare output directory path
    output_path = os.path.abspath(args.output_dir)
    output_train = os.path.join(output_path, "train")
    output_test = os.path.join(output_path, "test")

    # Save datasets
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(output_train):
        os.makedirs(output_train)

    if not os.path.exists(output_test):
        os.makedirs(output_test)

    df_train = df_X_train.merge(df_y_train, left_index=True, right_index=True)
    df_test = df_X_test.merge(df_y_test, left_index=True, right_index=True)
    df_train.to_csv(os.path.join(output_train, args.exp_name + ".csv"),
                    index_label="cellID", sep="\t", float_format='%.3f')
    df_test.to_csv(os.path.join(output_test, args.exp_name + ".csv"),
                   index_label="cellID", sep="\t", float_format='%.3f')


if __name__ == "__main__":
    main(sys.argv[1:])
