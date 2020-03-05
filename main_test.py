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
from sklearn.metrics import mean_absolute_error
from lib import utils
from model.ensemble_models import AveragingModels, StackingAveragedModels

# build
from lib.constant import features

if __name__ == "__main__":
    # run mode
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='random_search', type=str,
                        help='Run mode.')
    args = parser.parse_args()  

    ori_features = features
    input_features = []
    target_feature = 'PM2.5'

    # random search
    if args.mode == 'random_search':
        binary_features = np.random.randint(2, size=len(ori_features))
        times_random_search = 10
    elif args.mode == 'ones' or args.mode == 'one':
        binary_features = np.ones((len(ori_features),), dtype=int)
        times_random_search = 1
    elif args.mode == 'zeros' or args.mode == 'zero':
        binary_features = np.zeros((len(ori_features),), dtype=int)
        times_random_search = 1
    else:
        raise RuntimeError("Mode needs to be random_search/zeros/ones!")

    for time in range(1, 1+times_random_search):
        for index, value in enumerate(binary_features, start=0):
            if value == 1:
                input_features.append(ori_features[index])

        taiwan_dataset = pd.read_csv('data/csv/taiwan_data_mean.csv', usecols=input_features+[target_feature])
        new_dataset = utils.data_preprocessing(taiwan_dataset, input_features, target_feature)
        X_train, y_train, X_valid, y_valid, X_test, y_test = utils.split_data(new_dataset, 0.6, 0.2)
        eval_set = [(X_valid, y_valid)]

        # get models
        print("--Starting get models--")
        models = utils.get_models("SVR", "Lasso", "ElasticNet", 
                                "KernelRidge", "GradientBoostingRegressor", 
                                "LGBMRegressor", "XGBRegressor", "RandomForestRegressor",
                                "DecisionTreeRegressor", "AdaBoostRegressor",
                                "MLPRegressor", "KNeighborsRegressor", "ExtraTreesRegressor")
        
        adaboost = models["AdaBoostRegressor"]
        decisionTree = models["DecisionTreeRegressor"]
        extraTree = models["ExtraTreesRegressor"]
        mlp = models["MLPRegressor"]
        ENet = models["ElasticNet"]
        GBoost = models["GradientBoostingRegressor"]
        KRR = models["KernelRidge"]
        lasso = models["Lasso"]
        kn = models["KNeighborsRegressor"]
        randomForest = models["RandomForestRegressor"]
        xgb = models["XGBRegressor"]
        lgb = models["LGBMRegressor"]
        print("--Done get models!--")

        # test each model
        filename = "metrics.csv"        
        
        xgb.fit(X_train, y_train, eval_metric="mae", eval_set=eval_set, verbose=False, early_stopping_rounds = 10)
        xgb_train_pred = xgb.predict(X_train)
        xgb_pred = xgb.predict(X_test)
        mae_xgb = mean_absolute_error(y_test, xgb_pred)
        path_xgboost = "log/xgboost/"
        utils.write_log(path_xgboost, filename, [mae_xgb], input_features)  
        
        # # stacking
        # stacked_averaged_models = StackingAveragedModels(base_models = (xgb, GBoost),
        #                                             meta_model = KRR)
        # stacked_averaged_models.fit(X_train, y_train)
        # stacked_train_pred = stacked_averaged_models.predict(X_train)
        # stacked_pred = stacked_averaged_models.predict(X_test)
        # mae_stacking = mean_absolute_error(y_test, stacked_pred)
        # path_stacked = "log/stacking/"
        # utils.write_log(path_stacked, filename, [mae_stacking], input_features) 

        # reset input_features       
        input_features = []
        binary_features = np.random.randint(2, size=len(features))            
        