from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np; np.random.seed(0)
import pandas as pd
from sklearn import preprocessing
from sklearn import utils
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# use feature importance for feature selection
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier, XGBRegressor, XGBRFRegressor
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
# load data

def feature_importances_xgboost(dataset, cols_feature):
    dataset = dataset.to_numpy()
    # split data into X and y
    X = dataset[:,0:(len(cols_feature))]
    Y = dataset[:,-1]
    # split data into train and test sets
    train_size = int(len(dataset)*0.8)
    X_train = X[0:train_size]
    y_train = Y[0:train_size]
    
    X_test = X[train_size:]
    y_test = Y[train_size:]

    # fit model on all training data
    model = XGBRegressor(learning_rate=0.01, max_depth=3, min_child_weight=1.5, n_estimators=10000, seed=42)
    model.fit(X_train, y_train)
    # plot feature importance
    for col,score in zip(cols_feature,model.feature_importances_):
        print(col,score)
    plot_importance(model)
    plt.show()
    # make predictions for test data and evaluate
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    mae = mean_absolute_error(y_test, predictions)
    print("MAE: %.2f" % (mae))
    # Fit model using each importance as a threshold
    thresholds = sort(model.feature_importances_)
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        print(select_X_train[0,:])
        # train model
        selection_model = XGBRegressor(learning_rate=0.01, max_depth=3, min_child_weight=1.5, n_estimators=10000, seed=42) 
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        mae = mean_absolute_error(y_test, predictions)
        print("Thresh=%.5f, n=%d, MAE: %.2f" % (thresh, select_X_train.shape[1], mae))

def feature_importances_random_forest():
    df_train = pd.read_csv("data/csv/predicted_data.csv")

    model = RandomForestRegressor()
    training_score_y = df_train['PM2.5'].copy()
    training_input = df_train.drop(['PM2.5'], axis=1).copy()


    print(utils.multiclass.type_of_target(training_input))
    print(utils.multiclass.type_of_target(training_score_y))

    model.fit(training_input, training_score_y)

    importances = model.feature_importances_
    #Sort it
    print ("Sorted Feature Importance:")
    sorted_feature_importance = sorted(zip(importances, list(training_input)), reverse=True)
    print(sorted_feature_importance)

def correlation(data, file_name, method):
    """ 
    3 methods: pearson, kendall, spearman
    """
    corrmat = data.corr(method=method)
    print(corrmat)
    corrmat.to_csv('data/csv/{}_corr_mat_{}.csv'.format(file_name, method), encoding='utf-8', index=True)

    # plot corr mat
    # plt.figure(figsize=(10,5))
    # sns.heatmap(corrmat, vmin=-1, vmax=1)
    # plt.show()

if __name__ == "__main__":
    cols_feature_comparison_data = ['TIME','AMB_TEMP','CO','NO','NO2','NOx','O3','RH','SO2','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR', 'PM10', 'PM2.5']    
    dataset_comparison = pd.read_csv('data/csv/full_comparison_data_mean.csv')
    # correlation(dataset_comparison,'comparison', 'spearman')
    feature_importances_xgboost(dataset_comparison, cols_feature_comparison_data)
    cols_feature_original_data = ['TIME','WIND_SPEED','WIND_DIR','TEMP','RH','BAROMETER','RADIATION','INNER_TEMP','PM10','PM2.5']    
    dataset_original = pd.read_csv('data/csv/full_original_data_mean.csv')
    # correlation(dataset_original,'original', 'spearman')
    feature_importances_xgboost(dataset_original, cols_feature_original_data)

