from pandas import read_csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def generate_mean_data():
    var = ['wind_speed', 'wind_dir', 'temp', 'rh', 'barometer', 'inner_temp', 'pm_10', 'pm_2.5', 'pm_1']
    monitoring_data = read_csv('data/original_monitoring.csv', usecols=var)
    print(monitoring_data.isnull().sum())
    # replace by median values 
    monitoring_data.fillna(monitoring_data.mean(), inplace=True)
    print(monitoring_data.isnull().sum())
    np.savez('data/mean_data.npz', monitoring_data = monitoring_data)

def generate_comparison_data_uncorr():
    cols = ['PM10', 'PM2.5', 'RH', 'WIND_DIREC', 'WIND_SPEED']
    comparison_data = read_csv('data/comparison_data.csv', usecols=[i for i in range(2,26)])
    # convert dataframe to numeric to define whether values are a number or not
    for i in range(24):
        comparison_data.iloc[:,i] = pd.to_numeric(comparison_data.iloc[:,i], errors='coerce')                
    # replace missing values by mean data
    comparison_data.fillna(comparison_data.mean(), inplace=True)
    comparison_data = comparison_data.to_numpy()
    new_data = np.zeros(shape=(int(len(comparison_data)/5 * 24), 5), dtype='float')
    index = -1
    for row in range(len(comparison_data)):
        if row%5 == 0:
            index += 1
        new_data[24*index:24*(index+1),row%5] = comparison_data[row]
    
    switch_data = np.zeros(shape=(int(len(comparison_data)/5 * 24), 5), dtype='float')
    switch_data[:,0:3] = new_data[:, 2:]
    switch_data[:,3:] = new_data[:, 0:2]
    np.savez('data/comparison_data.npz', monitoring_data = switch_data)
    

def generate_data():
    predicted_var = ['PM10', 'PM2.5']
    dependent_var = ['AMB_TEMP', 'RH', 'WD_HR', 'WIND_DIREC']
    cols = ['time', 'wind_speed', 'temp', 'inner_temp', 'pm_10', 'pm_2.5', 'pm_1']
    monitoring_data = read_csv('data/original_monitoring.csv', usecols=cols)
    monitoring_data['time'] = pd.to_datetime(monitoring_data['time'])
    monitoring_data['time'] = pd.to_timedelta(monitoring_data['time'])
    monitoring_data['time'] = monitoring_data['time'] / pd.offsets.Minute(1)
    monitoring_data.fillna(monitoring_data.mean(), inplace=True)
    np.savez('data/corr_data_radiation.npz', monitoring_data = monitoring_data)

def generate_comparison_data_corr():
    cols = ['time', 'AMB_TEMP', 'RH', 'WD_HR', 'WIND_DIREC', 'PM10', 'PM2.5']
    comparison_data = read_csv('data/full_comparison_data.csv', usecols=cols)
    comparison_data['time'] = pd.to_datetime(comparison_data['time'])
    comparison_data['time'] = pd.to_timedelta(comparison_data['time'])
    comparison_data['time'] = comparison_data['time'] / pd.offsets.Minute(1)
    np.savez('data/comparison_data_corr.npz', monitoring_data = comparison_data)

def generate_comparison_data_fi_xgboost():
    cols = ['AMB_TEMP', 'CO', 'O3', 'SO2', 'WS_HR', 'PM10', 'PM2.5']
    comparison_data = read_csv('data/full_comparison_data.csv', usecols=cols)
    comparison_data['time'] = pd.to_datetime(comparison_data['time'])
    comparison_data['time'] = pd.to_timedelta(comparison_data['time'])
    comparison_data['time'] = comparison_data['time'] / pd.offsets.Minute(1)
    np.savez('data/comparison_data_fi_xgboost.npz', monitoring_data = comparison_data)


if __name__ == "__main__":
    generate_comparison_data_fi_xgboost()
