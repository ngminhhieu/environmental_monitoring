from pandas import read_csv
import numpy as np

if __name__ == "__main__":
    var = ['wind_speed', 'wind_dir', 'temp', 'rh', 'barometer', 'inner_temp', 'pm_10', 'pm_2.5', 'pm_1']
    monitoring_data = read_csv('data/predicted_data.csv', usecols=var)

    monitoring_data = monitoring_data.to_numpy()
    data = np.concatenate((monitoring_data[:, 0:5], monitoring_data[:, 5:]), axis=1)
    np.savez('data/predicted_data_2.npz', monitoring_data = data)

