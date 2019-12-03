from sklearn.ensemble import RandomForestRegressor
from pandas import read_csv
import pandas as pd

if __name__ == "__main__":
    predicted_var = ['pm_10', 'pm_2.5', 'pm_1']
    # dependent_var = ['wind_speed','wind_dir','temp', 'barometer', 'inner_temp']
    dependent_var = ['wind_speed','wind_dir','temp', 'rh', 'barometer', 'inner_temp']
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