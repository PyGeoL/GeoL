"""
Script to create word2vec models, given a set of mapped POIs.
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


def mergeUA_CellVector(BASE_DIR_CITY, CITY_NAME, SIZE1, METRIC, SIZE2, S, WS, C):
    """
    Given a city name and a grid size,
    returns a DataFrame joning Cell Vectors and Urban Atlas data
    """
    if METRIC == 'distance':
            # input Cell Vectors data mapped to grid
        CELLVECTOR_COUNT = os.path.join(BASE_DIR_CITY, 'embeddings', CITY_NAME + "_gs" + str(
            SIZE1) + "_skip_" + METRIC + "_s" + str(S) + "_ws" + str(WS) + "_c" + str(C) + ".txt")
    else:

        # input Cell Vectors data mapped to grid
        CELLVECTOR_COUNT = os.path.join(BASE_DIR_CITY, 'embeddings', CITY_NAME + "_gs" + str(
            SIZE1) + "_skip_" + METRIC+"_" + str(SIZE2) + "_s" + str(S) + "_ws" + str(WS) + "_c" + str(C) + ".txt")

    # input UA data mapped to grid
    UA_COUNT = os.path.join(BASE_DIR_CITY, 'count',
                            CITY_NAME + "_ua_count_" + str(SIZE1) + ".csv")

    print("------------------------------------------------------")
    print("Analising: \n", CELLVECTOR_COUNT, '\n', UA_COUNT)

    # load cellvector Count
    cellvector_tessellation = pd.read_csv(
        CELLVECTOR_COUNT, sep='\t', header=None)
    cols = [int(i) for i in cellvector_tessellation.columns]
    cols[0] = 'cellID'
    cellvector_tessellation.columns = cols
    cellvector_tessellation.columns = list(map(
        lambda x: 'f_fs_' + str(x) if x != "cellID" else x, cellvector_tessellation.columns))
    cellvector_tessellation.head(2)

    # load UA mapped onto grid
    ua_tessellation = pd.read_csv(UA_COUNT)
    ua_tessellation.columns = list(
        map(lambda x: 't_'+x if x != "cellID" else x, ua_tessellation.columns))

    # select only relevant columns
    ua_tessellation_target = ua_tessellation.loc[:, [
        'cellID', 't_predominant']]

    # Merge the UA and cellvector dataframes
    df_ua_fs = ua_tessellation_target.merge(
        cellvector_tessellation, on="cellID", how='left')
#     print('SPEREM----------------------------------------------------------')
#     print(df_ua_fs[df_ua_fs['cellID']==39269])

    return df_ua_fs


def split_train_test_from_file(BASE_DIR_CITY, CITY_NAME, SIZE, METRIC, S, WS, C):

    # output UA data mapped to grid
    OUTPUT_TRAIN, OUTPUT_TEST = [os.path.join(BASE_DIR_CITY, STEP, CITY_NAME + "_fs_" + str(SIZE) + "_skip_" + METRIC+"_" + str(
        SIZE) + "_s" + str(S) + "_ws" + str(WS) + "_c" + str(C) + ".csv") for STEP in ["train", "test"]]
    print(OUTPUT_TRAIN, OUTPUT_TEST)
    df_feat = pd.read_csv(OUTPUT_TRAIN, sep='\t')
    df_target = pd.read_csv(OUTPUT_TEST, sep='\t')

    # df_ua_fs.dropna(inplace=True)
    df_X_train = df_feat[[
        x for x in df_feat.columns if x.startswith('f_')]]
    df_X_test = df_target[[
        x for x in df_target.columns if x.startswith('f_')]]
    df_y_train = df_feat[[
        x for x in df_feat.columns if x.startswith('t_')]]
    df_y_test = df_target[[
        x for x in df_target.columns if x.startswith('t_')]]
    # # Divide train/test General
    # df_feat = df_ua_fs[[x for x in df_ua_fs.columns if x.startswith(
    #     'f_')]+['cellID']].set_index('cellID')
    # df_target = df_ua_fs[[x for x in df_ua_fs.columns if x.startswith(
    #     't_')]+['cellID']].set_index('cellID')

    # print("==================", df_ua_fs.shape, df_feat.shape, df_target.shape)
    # df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
    #     df_feat, df_target, test_size=0.2, random_state=42, stratify=df_target)

    return df_X_train, df_X_test, df_y_train, df_y_test


# def split_train_test(BASE_DIR_CITY, df_ua_fs, CITY_NAME, SIZE, METRIC, S, WS, C):

#     # output UA data mapped to grid
#     OUTPUT_TRAIN, OUTPUT_TEST = [os.path.join(BASE_DIR_CITY, STEP, CITY_NAME + "_fs_" + str(SIZE) + "_skip_" + METRIC+"_" + str(
#         SIZE) + "_s" + str(S) + "_ws" + str(WS) + "_c" + str(C) + ".csv") for STEP in ["train", "test"]]
#     OUTPUT_TRAIN_SCALED = '_scaled_.'.join(OUTPUT_TRAIN.split('.'))
#     OUTPUT_TEST_SCALED = '_scaled_.'.join(OUTPUT_TEST.split('.'))
#     print('adasdasd')
#     # print(OUTPUT_TRAIN, OUTPUT_TEST)
#     # print(OUTPUT_TRAIN_SCALED, OUTPUT_TEST_SCALED)

#     # Divide train/test General
#     df_feat = df_ua_fs[[x for x in df_ua_fs.columns if x.startswith(
#         'f_')]+['cellID']].set_index('cellID')
#     df_target = df_ua_fs[[x for x in df_ua_fs.columns if x.startswith(
#         't_')]+['cellID']].set_index('cellID')
#     df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
#         df_feat, df_target, test_size=0.2, random_state=42, stratify=df_target)

#     df_X_train.dropna(inplace=True)
#     df_X_test.dropna(inplace=True)
#     #     df_X_test.fillna(0, inplace=True)

#     # save datasets
#     df_train = df_X_train.merge(df_y_train, left_index=True, right_index=True)
#     df_test = df_X_test.merge(df_y_test, left_index=True, right_index=True)
#     df_train.to_csv(OUTPUT_TRAIN, index_label="cellID",
#                     sep="\t", float_format='%.6f')
#     df_test.to_csv(OUTPUT_TEST, index_label="cellID",
#                    sep="\t", float_format='%.6f')

#     # Create scaled version of train and test
#     X_scaler = StandardScaler()
#     df_X_train_scaled = X_scaler.fit_transform(df_X_train)
#     df_X_train_scaled = pd.DataFrame(
#         df_X_train_scaled, index=df_X_train.index, columns=df_X_train.columns)
#     df_X_test_scaled = X_scaler.transform(df_X_test)
#     df_X_test_scaled = pd.DataFrame(
#         df_X_test_scaled, index=df_X_test.index, columns=df_X_test.columns)

#     df_train_scaled = df_X_train_scaled.merge(
#         df_y_train, left_index=True, right_index=True)
#     df_test_scaled = df_X_test_scaled.merge(
#         df_y_test, left_index=True, right_index=True)

#     df_train_scaled.to_csv(OUTPUT_TRAIN_SCALED,
#                            index_label="cellID", sep="\t", float_format='%.6f')
#     df_test_scaled.to_csv(OUTPUT_TEST_SCALED,
#                           index_label="cellID", sep="\t", float_format='%.6f')

#     return df_train, df_test


def printEvalutationMetrics(df_y_test, y_pred):

    print(metrics.classification_report(df_y_test.values, y_pred))
    print("ACCURACY: {}".format(metrics.accuracy_score(df_y_test.values, y_pred)))
    print("F1 SCORE: {}".format(metrics.f1_score(
        df_y_test.values, y_pred, average='macro')))


def runExperiment(df_train, df_test, CITY_NAME, SIZE, BASE_DIR_CITY, SIZE1,  METRIC, S, WS, C):
    OUTPUT_PATH = os.path.join(BASE_DIR_CITY, "train")

    OUTPUT_FILENAME = os.path.join(
        OUTPUT_PATH, "metrics_s" + str(S) + "_ws" + str(WS) + "_c"+str(C)+".txt")

    dfs = []
    dim = 200
    df = {}  # {"area": boro, "cell": dim}

    suffix_train = "General"
    suffix_test = "General"

    df_y_train = df_train['t_predominant']
    df_y_test = df_test['t_predominant']

    # Baseline
    df_train['t_predominant'].value_counts().max()
    y_pred = [df_train['t_predominant'].value_counts().idxmax()] * \
        len(df_y_test)

    print("*****************************" + CITY_NAME +
          "  "+str(SIZE)+"*********************************")

    print("****** BASELINE ******")
    # Print Metrics
    printEvalutationMetrics(df_y_test, y_pred)
    df['model'] = "baseline_"+METRIC + "_s" + \
        str(S) + "_ws" + str(WS) + "_c"+str(C)

    # metrics.accuracy_score(df_y_test.values, y_pred)
    df['accuracy'] = metrics.accuracy_score(df_y_test.values, y_pred)
    # metrics.accuracy_score(df_y_test.values, y_pred)
    df['f1-score'] = metrics.f1_score(df_y_test.values,
                                      y_pred, average='macro')
    # metrics.accuracy_score(df_y_test.values, y_pred)
    df['precision'] = metrics.precision_score(
        df_y_test.values, y_pred, average='macro')
    # metrics.accuracy_score(df_y_test.values, y_pred)
    df['recall'] = metrics.recall_score(
        df_y_test.values, y_pred, average='macro')

    dfs.append(df)
    print("**********************")

    # # xgboost Classifier
    df = {}
    print("****** XGBOOST ******")
    df_X_train = df_train[[c for c in df_train.columns if c.startswith('f_')]]
    df_X_test = df_test[[c for c in df_test.columns if c.startswith('f_')]]

    # colsample_bytree=0.8, scale_pos_weight=1, learning_rate=0.1, min_child_weight=5,n_estimators=177, subsample=0.8, max_depth=3, gamma=0)
    clf = xgboost.XGBClassifier()
    clf.fit(df_X_train.as_matrix(), df_y_train.values)
    y_pred = clf.predict(df_X_test.as_matrix())
    # Print Metrics
    printEvalutationMetrics(df_y_test, y_pred)
    df['model'] = 'GBT_' + METRIC + "_s" + \
        str(S) + "_ws" + str(WS) + "_c"+str(C)

    # metrics.accuracy_score(df_y_test.values, y_pred)
    df['accuracy'] = metrics.accuracy_score(df_y_test.values, y_pred)
    # metrics.accuracy_score(df_y_test.values, y_pred)
    df['f1-score'] = metrics.f1_score(df_y_test.values,
                                      y_pred, average='macro')
    # metrics.accuracy_score(df_y_test.values, y_pred)
    df['precision'] = metrics.precision_score(
        df_y_test.values, y_pred, average='macro')
    # metrics.accuracy_score(df_y_test.values, y_pred)
    df['recall'] = metrics.recall_score(
        df_y_test.values, y_pred, average='macro')
    dfs.append(df)
    print(dfs)

    df = pd.DataFrame(dfs)
    print(df.head())

    with open(OUTPUT_FILENAME, 'a') as f:
        # Already has column names
        if (os.stat(OUTPUT_FILENAME).st_size > 0):
            df.to_csv(f, header=False, sep='\t')
        else:
            df.to_csv(f, header=True, sep='\t')

    print('********* CONFUSION MATRIX *******************')
    print(confusion_matrix(df_y_test.values, y_pred))

    print("********************************************************************************")


# ---------------------------  this functions serve for param estimation ------------------------------------


def modelfit(alg, X, y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, verbose=True):
    # print("XXXXXXXXXXXXXX", X)
    # print("yyyyyyyyyyyyyyyyyyy", y)

    le = preprocessing.LabelEncoder()
    y = list(le.fit_transform(y.values.ravel()))
    # print(len(le.fit_transform(y.values)), len(y), X.shape)
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X.values, label=y)
        # print(xgtrain.get_label())
        # print(le.fit_transform(y))
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()[
                          'n_estimators'], nfold=cv_folds, metrics='merror', early_stopping_rounds=early_stopping_rounds)  # verbose_eval=True)#show_progress=True)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(X, y, eval_metric='merror')

    if verbose:

        # Predict training set:
        predictions = alg.predict(X)

        # Print model report:
        print("\nModel Report")
        mse = metrics.mean_squared_error(y, predictions)
        print("MSE error (Train): %f" % mse)
        print("RMSE error (Train): %f" % math.sqrt(mse))


def tune(X, y, param_test, verbose=0, learning_rate=0.1, n_estimators=140, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, reg_alpha=0, seed=28, cv=5):

    print("THIS IS XXXXXXXXXXXXXXXXXX", X.iloc[:2, 1])
    print("THIS IS YYYYYYYYYYYYYYYYYYYY", y.iloc[:2, 0])

    gsearch = GridSearchCV(
        estimator=XGBClassifier(max_depth=max_depth,
                                learning_rate=learning_rate,
                                n_estimators=n_estimators,
                                silent=True,
                                objective='multi:softmax',
                                booster='gbtree',
                                n_jobs=1,
                                nthread=1,
                                gamma=gamma,
                                min_child_weight=min_child_weight,
                                max_delta_step=0,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                colsample_bylevel=1,
                                reg_alpha=reg_alpha,
                                reg_lambda=1,
                                scale_pos_weight=scale_pos_weight,
                                base_score=0.5,
                                random_state=0,
                                seed=seed,
                                missing=None),
        param_grid=param_test,
        scoring='f1_macro',
        n_jobs=2,
        iid=False,
        cv=cv,
        verbose=verbose)
    gsearch.fit(X, y)
    return gsearch.best_estimator_, gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
    # return gsearch.best_estimator_, gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_


def evaluate(alg, X_test, y_test):

    # print(alg.predict(X_test))
    # print(len(alg.predict(X_test)), X_test.shape, len(y_test))
    le = preprocessing.LabelEncoder()
    le.fit(y_test.values.ravel())
    # transform back encoded labels to strings ,e.g.:"Industrial"
    predictions = le.inverse_transform(alg.predict(X_test))
    # predictions = alg.predict(X_test)
    # print("NAAAAAAAAAAAAAAAAAAAAAAAAMESSSSSSSSSSSSSSSSSSSSSSSSSSS",
    #       y_test.values.ravel(), predictions)
    return sklearn.metrics.f1_score(y_test.values.ravel(), predictions, average="macro", labels=np.unique(y_test.values.ravel()))

    # return sklearn.metrics.f1_score(X_test, predictions, labels=y_test, average="macro", sample_weight=None)
    # sklearn.metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)
    # return math.sqrt(metrics.mean_squared_error(y_test, predictions))


def test_param(params, X_train, y_train, X_test, y_test, seed, verbose=True):
    # Costruisco un modello con i parametri specificati
    xgb1 = XGBClassifier(
        objective='multi:softmax',
        num_class=9,
        seed=seed)

    xgb1.set_params(**params)
    # Addestro il modello con una parte del dataset
    modelfit(xgb1, X_train, y_train, verbose=verbose)

    # Valuto il modello sul trainingset
    test_rmse = evaluate(xgb1, X_test, y_test)

    return xgb1, test_rmse


def build_model(X_train, y_train, X_test, y_test, seed, verbose=1):

    tuning = []
    testing = []

    def tune_and_update(param_test, parameters):

        best_estimator, grid_scores, best_params, best_score = tune(
            X_train, y_train, param_test, seed=seed, **parameters)

        if best_score >= tune_and_update.score:
            tune_and_update.score = best_score
            params.update(best_params)
            tuning.append((parameters.copy(), best_score))

        return best_score

    tune_and_update.score = float('-inf')

    # Inizializzo i parametri
    params = {}
    params['learning_rate'] = 0.1
    params['n_estimators'] = 1000
    params['max_depth'] = 5
    params['min_child_weight'] = 1
    params['gamma'] = 0
    params['subsample'] = 0.8
    params['colsample_bytree'] = 0.8
    params['scale_pos_weight'] = 1

    # Provo a costruire e valutare un primo modello con dei parametri iniziali
    alg, test_rmse = test_param(
        params, X_train, y_train, X_test, y_test, seed, verbose=verbose > 1)
    if verbose > 0:
        print('Primo modello\tTesting rmse = ' + str(test_rmse) + '\n')
    testing.append((params.copy(), test_rmse))

    # Inizio il tuning dei parametri

    params['n_estimators'] = 140

    param_test1 = {
        'max_depth': list(range(3, 10, 2)),
        'min_child_weight': list(range(1, 6, 2))
    }

    sc = tune_and_update(param_test1, params)
    if verbose > 0:
        print('Tuning 1\tScore = ' + str(sc))

    param_test2 = {
        'max_depth': [params['max_depth'] + k for k in [-1, 0, 1] if params['max_depth'] + k > 0],
        'min_child_weight': [params['min_child_weight'] + k for k in [-1, 0, 1] if params['min_child_weight'] + k > 0]
    }

    sc = tune_and_update(param_test2, params)
    if verbose > 0:
        print('Tuning 2\tScore = ' + str(sc))

    param_test2b = {
        'min_child_weight': [6, 8, 10, 12]
    }
    sc = tune_and_update(param_test2b, params)
    if verbose > 0:
        print('Tuning 2b\tScore = ' + str(sc))

    # Provo a valutare un modello con i parametri calcolati finora
    alg, test_rmse = test_param(
        params, X_train, y_train, X_test, y_test, seed, verbose=verbose > 1)
    if verbose > 0:
        print('Secondo modello\tTesting rmse = ' + str(test_rmse) + '\n')
    testing.append((params.copy(), test_rmse))

    # Continuo con il tuning

    param_test3 = {
        'gamma': [i/10.0 for i in range(0, 5)]
    }
    sc = tune_and_update(param_test3, params)
    if verbose > 0:
        print('Tuning 3\tScore = ' + str(sc))

    # Provo a valutare un modello con i parametri calcolati finora
    tmp_par = params.copy()
    tmp_par.update({'n_estimators': 1000})
    alg, test_rmse = test_param(
        tmp_par, X_train, y_train, X_test, y_test, seed, verbose=verbose > 1)
    if verbose > 0:
        print('Terzo modello\tTesting rmse = ' + str(test_rmse) + '\n')
    testing.append((tmp_par, test_rmse))

    # Continuo il tuning

    params['n_estimators'] = 177

    param_test4 = {
        'subsample': [i/10.0 for i in range(6, 10)],
        'colsample_bytree': [i/10.0 for i in range(6, 10)]
    }

    sc = tune_and_update(param_test4, params)
    if verbose > 0:
        print('Tuning 4\tScore = ' + str(sc))

    ss = int(params['subsample']*100)
    csbt = int(params['colsample_bytree']*100)
    param_test5 = {
        'subsample': [i/100.0 for i in range(max(0, ss-10), ss+5, 5)],
        'colsample_bytree': [i/100.0 for i in range(max(0, csbt-10), csbt+5, 5)]
    }
    sc = tune_and_update(param_test5, params)
    if verbose > 0:
        print('Tuning 5\tScore = ' + str(sc))

    param_test6 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    }
    sc = tune_and_update(param_test6, params)
    if verbose > 0:
        print('Tuning 6\tScore = ' + str(sc))

    if 'reg_alpha' in params:
        a = math.log10(params['reg_alpha'])
    else:
        a = 0
    param_test7 = {
        # [0, 0.001, 0.005, 0.01, 0.05]
        'reg_alpha': [0] + np.logspace(a-2, a+1, num=4)
    }

    sc = tune_and_update(param_test7, params)
    if verbose > 0:
        print('Tuning 7\tScore = ' + str(sc))

    '''
    # Provo a valutare il modello

    tmp_par = params.copy()
    tmp_par.update({'n_estimators' : 1000})
    alg, test_rmse = test_param(tmp_par, X_train, y_train, X_test, y_test, seed, verbose=verbose > 1)
    if verbose > 0:
        print ('Quarto modello\tTesting rmse = ' + str(test_rmse) + '\n')
    testing.append((tmp_par, test_rmse))

    # Provo un Learning Rate minore con un numero di Stimatori maggiore
    tmp_par = params.copy()
    tmp_par.update({'n_estimators' : 5000, 'learning_rate': 0.01})
    alg, test_rmse = test_param(tmp_par, X_train, y_train, X_test, y_test, seed, verbose=verbose > 1)
    if verbose > 0:
        print ('Quinto modello\tTesting rmse = ' + str(test_rmse) + '\n')
    testing.append((tmp_par, test_rmse))
    '''

    param_test8 = {
        'n_estimators': [10, 100, 1000, 3000],
        'learning_rate': [0.005, 0.01, 0.05, 0.1]
    }

    sc = tune_and_update(param_test8, params)
    if verbose > 0:
        print('Tuning 8\tScore = ' + str(sc))

    n = math.log10(params['n_estimators'])
    l = math.log10(params['learning_rate'])
    param_test9 = {
        'n_estimators': [int(x) for x in np.logspace(min(1, n-1), n+1, num=3)],
        'learning_rate': np.logspace(l-1, l+1, num=3)
    }

    sc = tune_and_update(param_test9, params)
    if verbose > 0:
        print('Tuning 9\tScore = ' + str(sc))

    return params, tuning, testing


def tuning_main_steps(CITY_NAME,  SIZE1,
                      BASE_DIR_CITY, SIZE2,  METRIC, S, WS, C):

    print('\tStarting ')

    X_train, X_test, y_train, y_test = split_train_test_from_file(
        BASE_DIR_CITY, CITY_NAME, SIZE1, METRIC, S, WS, C)

    print("X_train", len(X_train.values))
    print("y_train", len(y_train.values))
    print("X_test", len(X_test.values))
    print("y_test", len(y_test.values))

    print("X_train proporzione: ", len(X_train.values) /
          (len(X_train.values)+len(X_test.values)) * 100)
    print("X_test proporzione: ", len(X_test.values) /
          (len(X_train.values)+len(X_test.values)) * 100)
    print("y_train proporzione: ", len(y_train.values) /
          (len(y_train.values)+len(y_test.values)) * 100)
    print("y_test proporzione: ", len(y_test.values) /
          (len(y_train.values)+len(y_test.values)) * 100)

    params, tuning, testing = build_model(
        X_train, y_train, X_test, y_test, 27, verbose=1)

    print('\tValutazione modello:')

    alg, test_rmse = test_param(
        params, X_train, y_train, X_test, y_test, 27, verbose=1)
    print('\t\tTesting rmse = ')

    # std = data.std()
    # m = data.min()
    # M = data.max()
    # print('\t\tdata range = ' + str(M - m))
    # print('\t\tdata std = ' + str(std))
    # print('\t\trmse/std = ' + str(test_rmse/std))
    # print('\t\trmse/range = ' + str(test_rmse/(M - m)))

    # test_res = (test_rmse, std, M-m)

    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # if feat_imp.shape[0] > 0:
    #     feat_imp.plot(kind='bar', title='Feature Importances')
    #     plt.ylabel('Feature Importance Score')
    #     plt.savefig(os.path.join(outdir + '.pdf'), format="pdf", dpi=800, bbox_inches='tight')

    # with open(os.path.join(outdir, 'eval.csv'), 'w+') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    #     writer.writerow(['target', 'rmse', 'std', 'rng', 'rmse/std', 'rmse/rng'])

    #     for key, value in test_res.iteritems():
    #         rmse, std, rng = value
    #         writer.writerow([key, rmse, std, rng, rmse/std, rmse/rng])

    #     csvfile.close()

# -----------------------------------------------------------------------------------------------------------


def main(argv):

    parser = argparse.ArgumentParser('Run XGBOOST on Cellvector embeddings')

    parser.add_argument('-n', '--cityname',
                        help='CIty Name',
                        action='store',
                        dest='CITY_NAME',
                        required=True,
                        type=str)

    parser.add_argument('-s1', '--size-1',
                        help='Grid Size (from POI grid)',
                        action='store',
                        dest='SIZE1',
                        required=True,
                        type=int)

    parser.add_argument('-s2', '--size-2',
                        help='Grid Size (from word count step)',
                        action='store',
                        dest='SIZE2',
                        required=True,
                        type=int)

    parser.add_argument('-m', '--metric',
                        help='Metric employed(i.e. nearest, alphabetically, distance)',
                        action='store',
                        dest='METRIC',
                        required=True,
                        type=str)

    parser.add_argument('-s', '--w2v-size',
                        help='Word2Vec Gensim Size',
                        action='store',
                        dest='S',
                        required=True,
                        type=int)
    parser.add_argument('-ws', '--w2v-word-size',
                        help='Word2Vec Gensim Word Size',
                        action='store',
                        dest='WS',
                        required=True,
                        type=int)
    parser.add_argument('-c', '--w2v-count',
                        help='Word2Vec Gensim Count',
                        action='store',
                        dest='C',
                        required=True,
                        type=int)

    args = parser.parse_args()

    # Base directory
    BASE_DIR = os.path.abspath(".")
    # base directory for data files
    # BASE_DIR_DATA = os.path.join(BASE_DIR, "data")
    # base city
    BASE_DIR_CITY = os.path.join(BASE_DIR, 'data', args.CITY_NAME)

    # Tune parameters
    tuning_main_steps(args.CITY_NAME,  args.SIZE1,
                      BASE_DIR_CITY, args.SIZE1,  args.METRIC, args.S, args.WS, args.C)


if __name__ == "__main__":
    main(sys.argv[1:])
