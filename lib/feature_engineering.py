from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np; np.random.seed(0)
import pandas as pd
from sklearn import preprocessing
from sklearn import utils
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

def test_feature_importance():
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

def test_correlation(data, file_name):
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
    comparison_data = pd.read_csv('data/full_comparison_data.csv')
    print(comparison_data)
    test_correlation(comparison_data, 'comparison')