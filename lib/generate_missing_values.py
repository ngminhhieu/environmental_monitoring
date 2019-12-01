from sklearn.ensemble import RandomForestRegressor
from pandas import read_csv
import pandas as pd

if __name__ == "__main__":
    original_data = read_csv('data/original_monitoring.csv')
    for i in range(original_data.shape[0]):
        data = original_data.iloc[:,i+1]
        train_data = data[pd.isnull(data) == False]
        train_data_2d = train_data.values.reshape(train_data.shape[0],1)
        missing_data = data[pd.isnull(data)]
        missing_data_2d = missing_data.values.reshape(missing_data.shape[0],1)
        print(missing_data)
        # fill missing values
        rfModel= RandomForestRegressor()
        rfModel.fit(train_data_2d, train_data)

        generated_values = rfModel.predict(X = missing_data_2d)
        missing_data = generated_values.astype(float)
        data = train_data.append(missing_data)

    np.savez('data/predicted_data.npz', monitoring_data = data)
    