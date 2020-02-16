import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from lib import utils
from model.ensemble_models import AveragingModels, StackingAveragedModels

if __name__ == "__main__":
    
    # get dataset
    features = ['MONTH', 'DAY', 'YEAR', 'HOUR', 'AMB_TEMP', 'CO', 'NO', 'NO2',
    'NOx', 'O3', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR', 'PM10']
    # change later
    input_features = []

    # random search
    times_random_search = 1
    for time in range(1, 1+times_random_search):
        # find input_features by random search
        # binary_features = np.random.randint(2, size=len(features))
        binary_features = np.ones((17,), dtype=int)
        for index, value in enumerate(binary_features, start=0):
            if value == 1:
                input_features.append(features[index])
        print(binary_features)
        input_features.append('PM2.5')
        taiwan_dataset = pd.read_csv('data/csv/taiwan_data_mean.csv', usecols=input_features)
        X_train, y_train, X_valid, y_valid, X_test, y_test = utils.split_data(taiwan_dataset, input_features, 0.65, 0.15)
        eval_set = [(X_valid, y_valid)]
        
        # get models
        print("--Starting get models--")
        models = utils.get_models("SVR", "Lasso", "ElasticNet", 
                                "KernelRidge", "GradientBoostingRegressor", 
                                "LGBMRegressor", "XGBRegressor",
                                "DecisionTreeRegressor", "AdaBoostRegressor",
                                "MLPRegressor", "KNeighborsRegressor")
        
        decisionTree = models["DecisionTreeRegressor"]
        svr = models["SVR"]
        mlp = models["MLPRegressor"]
        ENet = models["ElasticNet"]
        GBoost = models["GradientBoostingRegressor"]
        KRR = models["KernelRidge"]
        lasso = models["Lasso"]
        model_xgb = models["XGBRegressor"]
        model_lgb = models["LGBMRegressor"]
        print("--Done get models!--")
        
        # model_xgb.fit(X_train, y_train, eval_metric="mae", eval_set=eval_set, verbose=False, early_stopping_rounds = 10)
        # xgb_train_pred = model_xgb.predict(X_train)
        # xgb_pred = model_xgb.predict(X_test)
        # mae_xgb = utils.mae(y_test, xgb_pred)
        # # write log
        # path_xgboost = "log/xgboost/"
        # utils.write_log(path_xgboost, input_features, [mae_xgb])  
        

        # model_lgb.fit(X_train, y_train, eval_metric="mae", eval_set=eval_set, verbose=False, early_stopping_rounds = 10)
        # lgb_train_pred = model_lgb.predict(X_train)
        # lgb_pred = model_lgb.predict(X_test)
        # mae_lgb = mean_absolute_error(y_test, lgb_pred)
        # # write log
        # path_lgb = "log/lgb/"
        # utils.write_log(path_lgb, input_features, [mae_lgb]) 


        stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                    meta_model = lasso)
        
        stacked_averaged_models.fit(X_train, y_train)
        stacked_train_pred = stacked_averaged_models.predict(X_train)
        stacked_pred = stacked_averaged_models.predict(X_test)
        mae_stacking = mean_absolute_error(y_test, stacked_pred)
        # write log
        path_stacked = "log/stacking/"
        utils.write_log(path_stacked, input_features, [mae_stacking]) 
        # reset input_features       
        input_features = []
    

    ### Stacking 
    # Avergage models
    averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

    # score = utils.mae_cv(averaged_models, X_train, y_train) 
    # print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    # Stacked average models
    
    # score = utils.mae_cv(stacked_averaged_models, X_train, y_train)
    # print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


   

    # '''MAE on the entire Train data when averaging'''

    # print('MAE score on train data:')
    # print(mean_absolute_error(y_train,stacked_train_pred*0.70 +
    # xgb_train_pred*0.15 + lgb_train_pred*0.15 ))

    # ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15