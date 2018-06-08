"""
Script to create word2vec models, given a set of mapped POIs.
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>
#          Michele Ferretti <mic.ferretti@gmail.com>

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


import numpy as np
import seaborn as sns

sns.set_style("ticks")
sns.set_context("paper")


import sklearn
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV  # Perforing grid search


from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import joblib
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 4

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
    clf.fit(df_X_train.as_matrix(), df_y_train.values.ravel())
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

def modelfit(model, X, y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, verbose=False):

    if useTrainCV:
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(X.values, label=y)
        cvresult = xgb.cv(xgb_param, xgtrain,
                          num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='merror', early_stopping_rounds=early_stopping_rounds)  # verbose_eval=True)#show_progress=True)
        model.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    model.fit(X, y, eval_metric='merror')

    if verbose:

        # Predict training set:
        predictions = model.predict(X)

        # Print model report:
        print("\n Model Report")
        mse = metrics.mean_squared_error(y, predictions)
        print("MSE error (Train): %f" % mse)
        print("RMSE error (Train): %f" % math.sqrt(mse))


def tune(X, y, param_test, verbose=0, learning_rate=0.1, n_estimators=140, max_depth=5, min_child_weight=1, gamma=0,
         subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, reg_alpha=0, seed=28, cv=5):

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


def evaluate(model, X_test, y_test):

    # transform back encoded labels to strings ,e.g.:"Industrial"
    predictions = model.predict(X_test)
    return sklearn.metrics.f1_score(y_test.values, predictions, average="macro"), predictions


def train_test(params, X_train, y_train, X_test, y_test, seed, verbose=True):

    num_class = len(y_train.drop_duplicates())

    model = XGBClassifier(objective='multi:softmax', num_class=num_class, seed=seed)
    model.set_params(**params)

    # Train and test the model
    modelfit(model, X_train, y_train, verbose=verbose)

    score, predictions = evaluate(model, X_test, y_test)

    return model, score, predictions


# ---------------------------  END  param estimation ------------------------------------


# TUNING AND TESTING
def build_model_and_tune(tuning, params, X_train, y_train, seed, verbose=1):

    # Best score and update of the parameters
    def tune_and_update(param_test, parameters):

        best_estimator, grid_scores, best_params, best_score = tune(X_train, y_train, param_test, seed=seed, **parameters)

        if best_score >= tune_and_update.score:
            tune_and_update.score = best_score
            params.update(best_params)
            tuning.append((parameters.copy(), best_score))

        return best_score

    tune_and_update.score = float('-inf')

    # Build a model with initial parameters
    #alg, f1_score, predictions = test_param(params, X_train, y_train, X_test, y_test, seed, verbose=verbose > 1)

    #if verbose > 0:
    #    print('Primo modello\tTesting rmse = ' + str(f1_score) + '\n')
    #testing.append((params.copy(), f1_score))

    # Tuning of the parameters
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

    param_test2b = {'min_child_weight': [6, 8, 10, 12]}
    sc = tune_and_update(param_test2b, params)

    if verbose > 0:
        print('Tuning 2b\tScore = ' + str(sc))

    param_test3 = {'gamma': [i/10.0 for i in range(0, 5)]}
    sc = tune_and_update(param_test3, params)

    if verbose > 0:
        print('Tuning 3\tScore = ' + str(sc))

    """
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
        'reg_alpha': [0] + np.logspace(a-2, a+1, num=4) # [0, 0.001, 0.005, 0.01, 0.05]
    }
    sc = tune_and_update(param_test7, params)

    if verbose > 0:
        print('Tuning 7\tScore = ' + str(sc))

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
    """

    return params, tuning


# NO TUNING JUST TRAIN+TESTING
def build_model(params, X_train, y_train, X_test, y_test, seed, verbose=1):

    model, score, predictions = train_test(params, X_train, y_train, X_test, y_test, seed, verbose=verbose > 1)

    return model, predictions, score

# -----------------------------------------------------------------------------------------------------------


def main(argv):

    parser = argparse.ArgumentParser('Run XGBOOST on Cellvector embeddings')

    parser.add_argument('-itr', '--input-train',
                        help='Input train',
                        action='store',
                        dest='input_train',
                        required=True,
                        type=str)

    parser.add_argument('-ite', '--input_test',
                        help='Input test',
                        action='store',
                        dest='input_test',
                        required=True,
                        type=str)

    parser.add_argument('-dm', '--directory-model',
                        help='Directory to store outputted model',
                        action='store',
                        dest='directory_model',
                        required=True,
                        type=str)

    parser.add_argument('-dp', '--directory-predictions',
                        help='Directory to store outputted predictions',
                        action='store',
                        dest='directory_predictions',
                        required=True,
                        type=str)

    parser.add_argument('-t', '--tuning',
                        help='Enable XGB parameter tuning. Disabled by default',
                        dest='enable_tuning',
                        action='store_true',
                        default=False)

    args = parser.parse_args()
    model_path = os.path.join(args.directory_model, 'test.model')
    pred_path = os.path.join(args.directory_predictions, 'pred.dat')

    # Load TRAIN data
    df_train = pd.read_csv(args.input_train, sep="\t")

    # Load TEST data
    df_test = pd.read_csv(args.input_test, sep="\t")

    le = preprocessing.LabelEncoder()
    labels = le.fit(df_train["target"].values.ravel())

    df_train["encoded_target"] = labels.transform(df_train["target"].values.ravel())
    df_test["encoded_target"] = labels.transform(df_test["target"].values.ravel())

    # Create Train/Test from dataframe
    X_train = df_train[[c for c in df_train.columns if c.startswith("f_")]]
    y_train = df_train["encoded_target"].values.ravel()

    X_test = df_test[[c for c in df_test.columns if c.startswith("f_")]]
    y_test = df_test["encoded_target"].values.ravel()

    # Check data and Train/Test proportions
    print("X_train", len(X_train.values))
    print("y_train", len(y_train))
    print("X_test", len(X_test.values))
    print("y_test", len(y_test))
    print("X_train proportions: ", len(X_train.values) /
          (len(X_train.values)+len(X_test.values)) * 100)
    print("X_test proportions: ", len(X_test.values) /
          (len(X_train.values)+len(X_test.values)) * 100)
    print("y_train proportions: ", len(y_train) /
          (len(y_train)+len(y_test)) * 100)
    print("y_test proportions: ", len(y_test) /
          (len(y_train)+len(y_test)) * 100)

    # Initialize variable for later use
    tuning = []

    # Initialize model parameters
    params = {}
    params['learning_rate'] = 0.1
    params['n_estimators'] = 1000
    params['max_depth'] = 5
    params['min_child_weight'] = 1
    params['gamma'] = 0
    params['subsample'] = 0.8
    params['colsample_bytree'] = 0.8
    params['scale_pos_weight'] = 1

    # If Tuning...
    if args.enable_tuning:
        # Train + Tune  parameters + Test
        params, tuning = build_model_and_tune(tuning, params, X_train, y_train, 27)

    else:
        # Train + Test
        params, tuning = build_model(tuning, params, X_train, y_train, 27)

    print('\tValutazione modello finale:')
    model, score, predictions = train_test(params, X_train, y_train, X_test, y_test, 27)

    # save_model(alg, args.directory_model)
    #joblib.dump(alg, MODEL_PATH)

    # save predictions
    pred_series = pd.Series(predictions)
    pred_series.to_csv(pred_path,index=None, header=False)

    # print('\t\tTesting rmse = ')
    print("----TUNING----\n")
    print(tuning)


    # TODO: ERA cosi, io l'ho cambiato sotto va bene? data = X.merge(Y, on=keys)
    # ma devi anche iterare su targets????

    #data = pd.concat([X_train, y_train])

    #std = data.std()
    #m = data.min()
    #M = data.max()
    #print('\t\tdata range = ' + str(M - m))
    #print('\t\tdata std = ' + str(std))
    #print('\t\trmse/std = ' + str(f1_score/std))
    #print('\t\trmse/range = ' + str(f1_score/(M - m)))

    #test_res = (f1_score, std, M-m)

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


if __name__ == "__main__":
    main(sys.argv[1:])
