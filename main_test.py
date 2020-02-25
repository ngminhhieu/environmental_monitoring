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
        X_train, y_train, X_valid, y_valid, X_test, y_test = utils.split_data(new_dataset, 0.65, 0.15)
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

        KRR.fit(X_train, y_train)
        KRR_pred = KRR.predict(X_test)
        mae_KRR = mean_absolute_error(y_test, KRR_pred)
        path_KRR = "log/KRR/"
        utils.write_log(path_KRR, filename, [mae_KRR], input_features)

        mlp.fit(X_train, y_train)
        mlp_pred = mlp.predict(X_test)
        mae_mlp = mean_absolute_error(y_test, mlp_pred)
        path_mlp = "log/mlp/"
        utils.write_log(path_mlp, filename, [mae_mlp], input_features)

        GBoost.fit(X_train, y_train)
        GBoost_pred = GBoost.predict(X_test)
        mae_GBoost = mean_absolute_error(y_test, GBoost_pred)
        path_GBoost = "log/GBoost/"
        utils.write_log(path_GBoost, filename, [mae_GBoost], input_features)
        
        xgb.fit(X_train, y_train, eval_metric="mae", eval_set=eval_set, verbose=False, early_stopping_rounds = 10)
        xgb_train_pred = xgb.predict(X_train)
        xgb_pred = xgb.predict(X_test)
        mae_xgb = mean_absolute_error(y_test, xgb_pred)
        path_xgboost = "log/xgboost/"
        utils.write_log(path_xgboost, filename, [mae_xgb], input_features)  
        
        # stacking
        stacked_averaged_models = StackingAveragedModels(base_models = (xgb, GBoost),
                                                    meta_model = KRR)
        stacked_averaged_models.fit(X_train, y_train)
        stacked_train_pred = stacked_averaged_models.predict(X_train)
        stacked_pred = stacked_averaged_models.predict(X_test)
        mae_stacking = mean_absolute_error(y_test, stacked_pred)
        path_stacked = "log/stacking/"
        utils.write_log(path_stacked, filename, [mae_stacking], input_features) 

        # averaged_models
        # averaged_models = AveragingModels(models = (GBoost, xgb, randomForest, extraTree))
        # averaged_models.fit(X_train, y_train)
        # averaged_model_train_pred = averaged_models.predict(X_train)
        # averaged_model_pred = averaged_models.predict(X_test)
        # mae_averaged_model = mean_absolute_error(y_test, averaged_model_pred)
        # path_averaged_model = "log/averaged_model/metrics.csv"
        # utils.write_log(path_averaged_model, filename, [mae_averaged_model], input_features) 

        # # reset input_features       
        # input_features = []
        # binary_features = np.random.randint(2, size=len(features))            
        