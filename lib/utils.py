import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime
from sklearn.metrics import mean_absolute_error

# models
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# support for models
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

# plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

error_invalid_model = "Invalid Models"
random_state = 42
# Cross validation
def mae_cv(model, X_train, Y_train, n_folds = 10):
    kf = KFold(n_folds, shuffle=True, random_state=random_state).get_n_splits(X_train)
    mae= -cross_val_score(model, X_train, Y_train, scoring="neg_mean_absolute_error", cv=kf)
    return(mae)

def split_data(dataset, train_per, valid_per):
    """dataset is a numpy array get from function data_preprocessing()"""

    # split data into X and y
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]
    # split data into train and test sets
    train_size = int(len(dataset)*train_per)
    valid_size = int(len(dataset)*valid_per)
    X_train = X[0:train_size]
    y_train = Y[0:train_size]

    X_valid = X[train_size:train_size+valid_size]
    y_valid = Y[train_size:train_size+valid_size]
    
    X_test = X[train_size:train_size+valid_size:]
    y_test = Y[train_size:train_size+valid_size:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def test_models(X_train, Y_train):
    # Modeling step Test differents algorithms 
    regressors = []
    regressors.append(SVR())
    regressors.append(DecisionTreeRegressor(random_state=random_state))
    regressors.append(AdaBoostRegressor(DecisionTreeRegressor(random_state=random_state),random_state=random_state,learning_rate=0.1))
    regressors.append(RandomForestRegressor(random_state=random_state))
    regressors.append(ExtraTreesRegressor(random_state=random_state))
    regressors.append(GradientBoostingRegressor(random_state=random_state))
    regressors.append(XGBRegressor(random_state=random_state))
    regressors.append(MLPRegressor(max_iter=400, random_state=random_state))
    regressors.append(KNeighborsRegressor())
    # regressors.append(LogisticRegression(random_state = random_state))
    # regressors.append(LinearDiscriminantAnalysis())
    regressors.append(make_pipeline(RobustScaler(), Lasso(random_state=random_state)))
    regressors.append(make_pipeline(RobustScaler(), ElasticNet(random_state=random_state)))
    regressors.append(KernelRidge())

    cv_results = []
    for regressor in regressors :
        # print(regressor)
        cv_results.append(mae_cv(regressor, X_train, Y_train, 2))
        # cv_results.append(-cross_val_score(regressor, X_train, Y_train, scoring="neg_mean_absolute_error"))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVR","DecisionTree","AdaBoost",
    "RandomForest","ExtraTrees","GradientBoosting", "XGBoost", "MultipleLayerPerceptron","KNeighboors", "Lasso", "ElasticNet", "KernelRidge"]})
    # "LogisticRegression",
    # "LinearDiscriminantAnalysis"
    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    plt.show()

def switch_model(model):
    svr = SVR()
    decisionTree = DecisionTreeRegressor(random_state=random_state, criterion="mae")
    adaBoost = AdaBoostRegressor(random_state=random_state)
    extraTree = ExtraTreesRegressor(random_state=random_state)
    randomForest = RandomForestRegressor(random_state=random_state)

    GBRegressor = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state = random_state)

    xgbregressor = XGBRegressor(objective ='reg:squarederror', max_depth=8, n_estimators=1000, min_child_weight=300, colsample_bytree=0.8, 
    subsample=0.8, eta=0.3, seed=random_state)

    mlp = MLPRegressor()

    kneighbor = KNeighborsRegressor()

    lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=random_state))

    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=random_state))

    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

    lgbm = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

    

    switcher={
        "SVR": svr,
        "RandomForestRegressor": randomForest,
        "DecisionTreeRegressor": decisionTree,
        "AdaBoostRegressor": adaBoost,
        "ExtraTreesRegressor": extraTree,
        "GradientBoostingRegressor": GBRegressor,
        "XGBRegressor":xgbregressor,
        "MLPRegressor": mlp,
        "KNeighborsRegressor": kneighbor,
        "Lasso": lasso,
        "ElasticNet": ENet,
        "KernelRidge": KRR,
        "LGBMRegressor": lgbm
    }

    return switcher.get(model, error_invalid_model)

def get_models(*args):
    models = {}
    for model_name in args:
        model = switch_model(model_name)
        if model == error_invalid_model:
            continue
        else:
            models[model_name] = model
    
    return models

def write_log(path, input_feature = [], error):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    error.insert(0, dt_string)
    if os.path.exists(path) == False:
        os.makedirs(path)
    with open(path + "metrics.csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(error)
        writer.writerow(input_feature)

def data_preprocessing(original_dataset, input_feature, target_feature, l=48, h=1):
    target_data = original_dataset.loc[:, target_feature]
    target_data_np = target_data.to_numpy()
    len_dataset = len(target_data)

    # hour_feature
    hour_feature = np.zeros(shape=(len_dataset-l, l))
    
    for i in range(len_dataset-l):
        hour_feature[i, 0:l] = target_data_np[i:i+l]
    
    # # statistical features
    # mean_feature = np.mean(hour_feature, axis=1)
    # mean_feature = mean_feature.reshape(len(mean_feature), 1)

    # min_feature = np.min(hour_feature, axis=1)
    # min_feature = min_feature.reshape(len(min_feature), 1)

    # max_feature = np.max(hour_feature, axis=1)
    # max_feature = max_feature.reshape(len(max_feature), 1)
    # statistical_feature = np.hstack([mean_feature, min_feature, max_feature])

    # drop first l rows
    dataset = original_dataset.loc[:, input_feature]
    dataset = dataset.to_numpy()
    dataset = dataset[l:, ]

    # drop first l rows of target_feature
    target_data_np = target_data_np[l:]
    target_data_np = target_data_np.reshape(len(target_data_np), 1)


    # merge hour features
    """Columns: Input Features - Hour Features - Target Feature"""
    new_dataset = np.concatenate([dataset, hour_feature, target_data_np], axis=1)
    
    new_dataset = np.around(new_dataset, decimals=1)
    
    return new_dataset
        
    
