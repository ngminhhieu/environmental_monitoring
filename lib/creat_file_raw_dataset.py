from pandas import read_csv
import numpy as np

if __name__ == "__main__":
    var = ['wind_speed', 'wind_dir', 'temp', 'barometer', 'inner_temp', 'pm_10', 'pm_2.5', 'pm_1']
    monitoring_data = read_csv('data/original_monitoring.csv', usecols=var)
    # print(monitoring_data.isnull().sum())

    # replace by median values
    median = monitoring_data['wind_speed'].median()
    print(median)
    monitoring_data['wind_speed'].fillna(median, inplace=True)
    median = monitoring_data['wind_dir'].median()
    print(median)
    monitoring_data['wind_dir'].fillna(median, inplace=True) 
    median = monitoring_data['temp'].median()
    print(median)
    monitoring_data['temp'].fillna(median, inplace=True) 
    median = monitoring_data['barometer'].median()
    print(median)
    monitoring_data['barometer'].fillna(median, inplace=True) 
    median = monitoring_data['inner_temp'].median()
    print(median)
    monitoring_data['inner_temp'].fillna(median, inplace=True) 
    median = monitoring_data['pm_10'].median()
    monitoring_data['pm_10'].fillna(median, inplace=True) 
    median = monitoring_data['pm_2.5'].median()
    monitoring_data['pm_2.5'].fillna(median, inplace=True)
    median = monitoring_data['pm_1'].median()
    monitoring_data['pm_1'].fillna(median, inplace=True) 

    monitoring_data = monitoring_data.to_numpy()
    data = np.concatenate((monitoring_data[:, 0:5], monitoring_data[:, 5:]), axis=1)
    np.savez('data/mean_data.npz', monitoring_data = data)

    # predicted_data = read_csv('data/predicted_data.csv')
    # np.savez('data/predicted_data.npz', monitoring_data = predicted_data)

