from pandas import read_csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def mean_data():
    var = ['wind_speed', 'wind_dir', 'temp', 'rh', 'barometer', 'inner_temp', 'pm_10', 'pm_2.5', 'pm_1']
    monitoring_data = read_csv('data/original_monitoring.csv', usecols=var)
    print(monitoring_data.isnull().sum())
    # replace by median values 
    monitoring_data.fillna(monitoring_data.mean(), inplace=True)
    print(monitoring_data.isnull().sum())
    # np.savez('data/mean_data.npz', monitoring_data = monitoring_data)


def random_forest_data():
    predicted_var = ['pm_10', 'pm_2.5', 'pm_1']
    dependent_var = ['wind_speed', 'wind_dir', 'temp', 'rh', 'barometer', 'inner_temp']
    original_data = read_csv('data/original_monitoring.csv')
    
    for i in range(len(predicted_var)):
        with_pm = original_data[pd.isnull(original_data[predicted_var[i]]) == False]
        without_pm = original_data[pd.isnull(original_data[predicted_var[i]])]
        # fill missing values
        rfModel= RandomForestRegressor()
        rfModel.fit(with_pm[dependent_var], with_pm[predicted_var[i]])

        generated_values = rfModel.predict(X = without_pm[dependent_var])
        without_pm[predicted_var[i]] = generated_values.astype(float)
        data = with_pm.append(without_pm)
        data.sort_index(inplace=True)
        original_data[predicted_var[i]] = data[predicted_var[i]]
        original_data.to_csv('data/predicted_data_2.csv', encoding='utf-8', index=False)
    
    np.savez('data/predicted_data_2.npz', monitoring_data = original_data)

def comparison_data():
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
    

def test():
    var = ['wind_speed', 'wind_dir', 'temp', 'rh', 'barometer', 'inner_temp', 'pm_10', 'pm_2.5', 'pm_1']
    monitoring_data = read_csv('data/original_monitoring.csv', usecols=var)
    print(monitoring_data['pm_10'].isnull().sum())

if __name__ == "__main__":
    # mean_data()
    # random_forest_data()
    comparison_data()
    # test()
