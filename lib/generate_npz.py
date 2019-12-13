from pandas import read_csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def generate_original_data():
    cols = ['TIME','WIND_SPEED','WIND_DIR','TEMP','RH','BAROMETER','RADIATION','INNER_TEMP','PM2.5']
    dataset = read_csv('data/csv/full_original_data_mean.csv')
    np.savez('data/npz/original_data_mean.npz', monitoring_data = dataset)

def generate_comparison_data():
    cols = ['WIND_DIREC', 'WIND_SPEED', 'AMB_TEMP','RH','PM10', 'PM2.5']
    dataset = read_csv('data/csv/full_comparison_data_mean.csv', usecols=cols)
    np.savez('data/npz/comparison_data_mean.npz', monitoring_data = dataset)

def generate_original_data_fi_xgboost():
    cols = ['TIME','WIND_SPEED','WIND_DIR','TEMP','RH','BAROMETER','RADIATION','INNER_TEMP','PM2.5']
    data = read_csv('data/csv/full_original_data_mean.csv', usecols=cols)
    np.savez('data/npz/original_data_fi_xgboost_mean.npz', monitoring_data = data)

def generate_comparison_data_fi_xgboost():
    cols = ['AMB_TEMP', 'CO', 'O3', 'SO2', 'WS_HR', 'PM2.5']
    comparison_data = read_csv('data/csv/full_comparison_data_mean.csv', usecols=cols)
    np.savez('data/npz/comparison_data_fi_xgboost_mean.npz', monitoring_data = comparison_data)

def generate_original_data_corr():
    cols = ['TIME','WIND_SPEED','TEMP','INNER_TEMP','PM2.5']
    monitoring_data = read_csv('data/csv/full_original_data_mean.csv', usecols=cols)
    np.savez('data/npz/original_data_corr_mean.npz', monitoring_data = monitoring_data)

def generate_comparison_data_corr():
    # cols = ['time', 'AMB_TEMP', 'RH', 'WD_HR', 'WIND_DIREC', 'PM2.5']
    cols = ['AMB_TEMP', 'RH', 'PM2.5']
    comparison_data = read_csv('data/csv/full_comparison_data_mean.csv', usecols=cols)
    np.savez('data/npz/comparison_data_corr_mean.npz', monitoring_data = comparison_data)

if __name__ == "__main__":
    generate_original_data()
    generate_comparison_data()
    # generate_original_data_fi_xgboost()
    # generate_comparison_data_fi_xgboost()
    generate_original_data_corr()
    generate_comparison_data_corr()
