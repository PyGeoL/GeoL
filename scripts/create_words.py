
import os, errno
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import sys,getopt
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
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search


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

strategies ={ 1:'alphabetically',2:'nearest',3:'distance'}

CITIES = ['barcelona']

for CITY in CITIES: 
    BASE_DIR_CITY = os.path.join(BASE_DIR_DATA, CITY)   
    OUTPUT_FOLDER = os.path.join(BASE_DIR_DATA, CITY, 'words')    
  
    for strategy in strategies.items():

        if strategy[1] == 'distance':
            
            # foursquare raw
            INPUT_FILE = os.path.join(BASE_DIR_CITY, 'foursquare_raw', CITY +"_poi.csv") # WORKING WITH FAKE FS DATA
            print(OUTPUT_FOLDER, INPUT_FILE,S, sep )
            S = strategy[0]
            sep = "\",\""
            %run GeoL/scripts/create_sequences.py -o $OUTPUT_FOLDER -i $INPUT_FILE -s $S -sp $sep -v

        else:            

            for FILE in os.listdir(os.path.join(BASE_DIR_CITY, 'mapped')):
                SIZE = FILE.split('.')[0].split('_')[-1]
                INPUT_FILE = os.path.join(BASE_DIR_CITY, 'mapped', CITY +"_fs_grid_"+SIZE+".csv") # WORKING WITH FAKE FS DATA
                EMPTY_GRID = os.path.join(BASE_DIR_CITY,'grid','grid_square_'+SIZE+'.geojson') 
                # nearest
                if strategy[1] == 'nearest':
                        # foursquare raw
                        INPUT_FILE = os.path.join(BASE_DIR_CITY, 'mapped', CITY +"_fs_grid_"+SIZE+".csv") # WORKING WITH FAKE FS DATA
                        print(strategy[0], OUTPUT_FOLDER, INPUT_FILE)
                        S = strategy[0]
                        print(type(S))
                        sep = "\"\t\""
                        %run GeoL/scripts/create_sequences.py -o $OUTPUT_FOLDER -i $INPUT_FILE -g $EMPTY_GRID -s $S -sp $sep -gs $SIZE -v

                # alphabetically
                else:
                    INPUT_FILE = os.path.join(BASE_DIR_CITY, 'mapped', CITY +"_fs_grid_"+SIZE+".csv") # WORKING WITH FAKE FS DATA

                    print(strategy[0], OUTPUT_FOLDER, INPUT_FILE )
                    S = strategy[0]
                    print(type(S))
                    sep = "\"\t\""
                    %run GeoL/scripts/create_sequences.py -o $OUTPUT_FOLDER -i $INPUT_FILE -s $S -sp $sep -gs $SIZE -v



