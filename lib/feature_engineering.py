import numpy as np; np.random.seed(0)
import pandas as pd
from numpy import sort
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectFromModel

def feature_importances_xgboost(dataset, cols_feature, name):
    dataset = dataset.to_numpy()
    # split data into X and y
    X = dataset[:,0:(len(cols_feature)-2)]
    Y = dataset[:,-1]
    # split data into train and test sets
    train_size = int(len(dataset)*0.8)
    X_train = X[0:train_size]
    y_train = Y[0:train_size]
    
    X_test = X[train_size:]
    y_test = Y[train_size:]

    # fit model on all training data
    model = XGBRegressor(learning_rate=0.01, max_depth=3, min_child_weight=1.5, n_estimators=10000, seed=42)
    model.fit(X_train, y_train)
    # plot feature importance
    for col,score in zip(cols_feature,model.feature_importances_):
        print(col,score)
    feature_importances = list(zip(cols_feature,model.feature_importances_))
    feature_importances = sorted(feature_importances, key=lambda x: x[1])

    # make predictions for test data and evaluate
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    mae = mean_absolute_error(y_test, predictions)
    print("MAE: %.3f" % (mae))
    # this var to get the lowest mae 
    temp_mae = float("inf")
    # Fit model using each importance as a threshold
    for _, thresh in feature_importances:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = XGBRegressor(learning_rate=0.01, max_depth=3, min_child_weight=1.5, n_estimators=10000, seed=42) 
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        mae = mean_absolute_error(y_test, predictions)
        if mae < temp_mae:
            # get feature that being important
            features = [feature for (feature,threshold) in feature_importances if threshold >= thresh]
            if name == "taiwan_data":
                np.savez('data/npz/feature_engineering/taiwan_data_xgboost.npz', features = features)
            else:
                np.savez('data/npz/feature_engineering/hanoi_data_xgboost.npz', features = features)
        temp_mae = mae
        print("Thresh=%.3f, n=%d, MAE: %.3f" % (thresh, select_X_train.shape[1], mae))

if __name__ == "__main__":
    # taiwan
    cols_taiwan = ['TIME','AMB_TEMP','CO','NO','NO2','NOx','O3','RH','SO2','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR', 'PM10', 'PM2.5']    
    taiwan_dataset = pd.read_csv('data/csv/taiwan_data_mean.csv')
    feature_importances_xgboost(taiwan_dataset, cols_taiwan, 'taiwan_data')

    # ha noi
    cols_hanoi = ['TIME','WIND_SPEED','WIND_DIR','TEMP','RH','BAROMETER','RADIATION','INNER_TEMP','PM10','PM2.5']    
    hanoi_dataset = pd.read_csv('data/csv/hanoi_data_mean.csv')
    feature_importances_xgboost(hanoi_dataset, cols_hanoi, 'hanoi_data')

