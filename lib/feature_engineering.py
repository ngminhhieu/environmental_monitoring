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
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
# load data

def feature_importances_xgboost():
    dataset = pd.read_csv('data/full_comparison_data.csv')
    dataset['time'] = pd.to_datetime(dataset['time'])
    dataset['time'] = dataset['time'].values.astype(float)
    dataset = dataset.to_numpy()
    # split data into X and y
    X = dataset[:,0:13]
    Y = dataset[:,13]
    # split data into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    # fit model on all training data
    train_size = int(len(dataset)*0.8)
    X_train = X[0:train_size]
    y_train = Y[0:train_size]
    
    X_test = X[train_size:]
    y_test = Y[train_size:]

    model = XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 
    model.fit(X_train, y_train)
    # make predictions for test data and evaluate
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = mean_absolute_error(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # Fit model using each importance as a threshold
    thresholds = sort(model.feature_importances_)
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = mean_absolute_error(y_test, predictions)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

def feature_importances_random_forest():
    df_train = pd.read_csv("data/predicted_data.csv")

    df_train['time'] = pd.to_datetime(df_train['time'])
    df_train['time'] = pd.to_timedelta(df_train['time'])
    df_train['time'] = df_train['time'] / pd.offsets.Minute(1)

    model = RandomForestRegressor()
    training_score_y = df_train['pm_2.5'].copy()
    training_input = df_train.drop(['pm_2.5'], axis=1).copy()


    print(utils.multiclass.type_of_target(training_input))
    print(utils.multiclass.type_of_target(training_score_y))

    model.fit(training_input, training_score_y)

    importances = model.feature_importances_
    #Sort it
    print ("Sorted Feature Importance:")
    sorted_feature_importance = sorted(zip(importances, list(training_input)), reverse=True)
    print(sorted_feature_importance)

def correlation(data, file_name):
    data['time'] = pd.to_datetime(data['time'])
    data['time'] = data['time'].values.astype(float)
    corrmat_pearson = data.corr(method='pearson')
    # corrmat_pearson.to_csv('data/{}_corr_mat_pearson.csv'.format(file_name), encoding='utf-8', index=True)

    corrmat_kendall = data.corr(method='kendall')
    # corrmat_kendall.to_csv('data/{}_corr_mat_kendall.csv'.format(file_name), encoding='utf-8', index=True)
    print(corrmat_kendall)
    corrmat_spearman = data.corr(method='spearman')
    # corrmat_spearman.to_csv('data/{}_corr_mat_spearman.csv'.format(file_name), encoding='utf-8', index=True)

    plt.figure(figsize=(10,5))
    sns.heatmap(corrmat_pearson, vmin=-1, vmax=1)
    plt.show()
    sns.heatmap(corrmat_kendall, vmin=-1, vmax=1)
    plt.show()
    sns.heatmap(corrmat_spearman, vmin=-1, vmax=1)
    plt.show()

if __name__ == "__main__":
    # comparison_data = pd.read_csv('data/full_comparison_data.csv')
    # print(comparison_data)
    # correlation(comparison_data, 'comparison')
    feature_importances_xgboost()