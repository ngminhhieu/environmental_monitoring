from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    monitoring_data = read_csv('data/original_monitoring.csv')
    # print(monitoring_data['time'].isnull().sum())
    # print(monitoring_data['wind_speed'].isnull().sum())
    # print(monitoring_data['wind_dir'].isnull().sum())
    # print(monitoring_data['temp'].isnull().sum())
    # print(monitoring_data['rh'].isnull().sum())
    # print(monitoring_data['barometer'].isnull().sum())
    # print(monitoring_data['radiation'].isnull().sum())
    # print(monitoring_data['inner_temp'].isnull().sum())
    # print(monitoring_data['pm_10'].isnull().sum())
    # print(monitoring_data['pm_2.5'].isnull().sum())
    # print(monitoring_data['pm_1'].isnull().sum())

    # replace by median values
    median = monitoring_data['wind_speed'].median()
    monitoring_data['wind_speed'].fillna(median, inplace=True)
    median = monitoring_data['wind_dir'].median()
    monitoring_data['wind_dir'].fillna(median, inplace=True) 
    median = monitoring_data['temp'].median()
    monitoring_data['temp'].fillna(median, inplace=True) 
    median = monitoring_data['rh'].median()
    monitoring_data['rh'].fillna(median, inplace=True) 
    median = monitoring_data['barometer'].median()
    monitoring_data['barometer'].fillna(median, inplace=True) 
    median = monitoring_data['radiation'].median()
    monitoring_data['radiation'].fillna(median, inplace=True) 
    median = monitoring_data['inner_temp'].median()
    monitoring_data['inner_temp'].fillna(median, inplace=True) 
    median = monitoring_data['pm_10'].median()
    monitoring_data['pm_10'].fillna(median, inplace=True) 
    median = monitoring_data['pm_2.5'].median()
    monitoring_data['pm_2.5'].fillna(median, inplace=True)
    median = monitoring_data['pm_1'].median()
    monitoring_data['pm_1'].fillna(median, inplace=True) 

    monitoring_data = monitoring_data.to_numpy()
    data = np.concatenate((monitoring_data[:, 1:5], monitoring_data[:, 8:]), axis=1)
    np.savez('data/modified_monitoring_data.npz', monitoring_data = data)
    dataset=np.load('data/modified_monitoring_data.npz')

