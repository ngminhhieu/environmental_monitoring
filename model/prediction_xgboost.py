import numpy as np; np.random.seed(0)
import pandas as pd
from numpy import sort
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def predict(dataset, cols_feature):
    # split data into X and y
    X = dataset.iloc[:,0:(len(cols_feature)-1)]
    Y = dataset.iloc[:,-1]
    # split data into train and test sets
    train_size = int(len(dataset)*0.8)
    X_train = X.iloc[0:train_size, :]
    y_train = Y.iloc[0:train_size, :]
    
    X_test = X.iloc[train_size:, :]
    y_test = Y.iloc[train_size:, :]

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
    verbose=True, 
    early_stopping_rounds = 10)
    # plot feature importance
    # feature_importances = pd.DataFrame({'col': columns,'imp':model.feature_importances_})
    # feature_importances = feature_importances.sort_values(by='imp',ascending=False)
    # px.bar(feature_importances,x='col',y='imp')

    # make predictions for test data and evaluate
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    mae = mean_absolute_error(y_test, predictions)
    print("MAE: %.3f" % (mae))
    

if __name__ == "__main__":
    # taiwan
    features = ['MONTH', 'DAY', 'YEAR', 'HOUR', 'AMB_TEMP', 'CO', 'NO', 'NO2',
    'NOx', 'O3', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR', 'PM10', 'PM2.5']    
    taiwan_dataset = pd.read_csv('data/csv/taiwan_data_mean.csv', usecols=features)
    predict(taiwan_dataset, features)

