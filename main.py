# ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# run mode
import sys
import os
import argparse

# other...
import pandas as pd
import numpy as np
from lib import constant
from sklearn.metrics import mean_absolute_error
from lib import utils
from model.ensemble_models import AveragingModels, StackingAveragedModels

# build
from lib.constant import features

if __name__ == "__main__":
    # args
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='random_search', type=str,
                        help='Run mode.')
    args = parser.parse_args()  
    
    input_features = constant.features
    target_feature = 'PM2.5'

    taiwan_dataset = pd.read_csv('data/csv/taiwan_data_mean.csv', usecols=input_features+[target_feature])
    new_dataset = utils.data_preprocessing(taiwan_dataset, input_features, target_feature)
    X_train, y_train, X_valid, y_valid, X_test, y_test = utils.split_data(new_dataset, 0.65, 0.15)
    eval_set = [(X_valid, y_valid)]
    utils.test_models(X_train, y_train)
      
        