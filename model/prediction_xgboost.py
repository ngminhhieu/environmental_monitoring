import numpy as np;
import pandas as pd
import csv
from datetime import datetime
from numpy import sort
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import plotly_express as px
import os

def predict(dataset, cols_feature):
    # split data into X and y
    X = dataset.iloc[:,0:(len(cols_feature)-1)]
    Y = dataset.iloc[:,-1]
    # split data into train and test sets
    train_size = int(len(dataset)*0.65)
    valid_size = int(len(dataset)*0.15)
    X_train = X.iloc[0:train_size]
    y_train = Y.iloc[0:train_size]

    X_valid = X.iloc[train_size:train_size+valid_size]
    y_valid = Y.iloc[train_size:train_size+valid_size]
    
    X_test = X.iloc[train_size:train_size+valid_size:]
    y_test = Y.iloc[train_size:train_size+valid_size:]
    eval_set = [(X_valid, y_valid)]
    
    # fit model on all training data
    model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)
    model.fit(
    X_train, 
    y_train, 
    eval_metric="mae",
    eval_set=eval_set,
    verbose=False, 
    early_stopping_rounds = 10)

    # plot feature importance
    feature_importances = pd.DataFrame({'col': cols_feature[0:-1],'imp':model.feature_importances_})
    feature_importances = feature_importances.sort_values(by='imp',ascending=False)
    px.bar(feature_importances, x='col',y='imp')

    # make predictions for test data and evaluate
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    mae = mean_absolute_error(y_test, predictions)
    print("MAE: %.3f" % (mae))
    return [mae]

def write_log(path, input_features, error):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    error.insert(0, dt_string)
    if os.path.exists(path) == False:
        os.makedirs(path)
    with open(path + "metrics.csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(error)
        writer.writerow(input_features)

if __name__ == "__main__":
    # taiwan
    features = ['MONTH', 'DAY', 'YEAR', 'HOUR', 'AMB_TEMP', 'CO', 'NO', 'NO2',
    'NOx', 'O3', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR', 'PM10']    
    input_features = []

    # random search
    times_random_search = 1    
    for time in range(1, 1+times_random_search):
        # find input_features by random search
        binary_features = np.random.randint(2, size=len(features))
        for index, value in enumerate(binary_features, start=0):
            if value == 1:
                input_features.append(features[index])
        print(binary_features)
        input_features.append('PM2.5')
        taiwan_dataset = pd.read_csv('data/csv/taiwan_data_mean.csv', usecols=input_features)
        error_mae = predict(taiwan_dataset, input_features)

        # reset input_features
        path = "log/xgboost/"
        write_log(path, input_features, error_mae)        
        input_features = []
        
    
        