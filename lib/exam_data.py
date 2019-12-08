import datetime
from pandas import read_csv
from scipy.stats import spearmanr, pearsonr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def check_correlation():
    var = ['time','wind_speed','wind_dir','temp','rh','barometer','radiation','inner_temp','pm_10','pm_2.5','pm_1']
    corr_mat = np.zeros(shape=(len(var),len(var)))
    data = read_csv('data/predicted_data.csv')
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            corr, _ = spearmanr(data[var[i]], data[[var[j]]])
            corr_mat[i][j] = corr
    
    np.savetxt('data/corr_mat.csv', corr_mat, delimiter=',')

def random_forest_data():
    predicted_var = ['pm_10', 'pm_2.5', 'pm_1']
    dependent_var = ['time', 'wind_speed', 'temp', 'inner_temp']
    original_data = read_csv('data/original_monitoring.csv', usecols=dependent_var + predicted_var)
    original_data['time'] = pd.to_datetime(original_data['time'])
    original_data['time'] = pd.to_timedelta(original_data['time'])
    original_data['time'] = original_data['time'] / pd.offsets.Minute(1)
    
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
        original_data.to_csv('data/uncorr_test_data.csv', encoding='utf-8', index=False)
    
    np.savez('data/uncorr_test_data.npz', monitoring_data = original_data)

if __name__ == "__main__":
    random_forest_data()