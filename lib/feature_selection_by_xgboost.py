import numpy as np; np.random.seed(0)
import pandas as pd
from numpy import sort
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import constant

def feature_importances_xgboost(dataset, cols_feature, train_per=0.2, valid_per=0.2):
    dataset = dataset.to_numpy()
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]
    # split data into train and test sets
    train_size = int(len(dataset)*train_per)
    valid_size = int(len(dataset)*valid_per)
    X_train = X[0:train_size]
    y_train = Y[0:train_size]

    X_valid = X[train_size:train_size+valid_size]
    y_valid = Y[train_size:train_size+valid_size]

    X_test = X[train_size:train_size+valid_size:]
    y_test = Y[train_size:train_size+valid_size:]



    model = XGBRegressor(objective ='reg:squarederror', max_depth=8, n_estimators=1000, min_child_weight=300, colsample_bytree=0.8, 
    subsample=0.8, eta=0.3, seed=2)

    model.fit(X_train, y_train, eval_metric="mae", eval_set=[(X_valid, y_valid)], verbose=False, early_stopping_rounds = 15)
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
        select_X_valid = selection.transform(X_valid)
        select_X_test = selection.transform(X_test)
        # train model
        selection_model = XGBRegressor(objective ='reg:squarederror', max_depth=8, n_estimators=1000, min_child_weight=300, colsample_bytree=0.8, 
    subsample=0.8, eta=0.3, seed=2)
        selection_model.fit(select_X_train, y_train, eval_metric="mae", eval_set=[(select_X_valid, y_valid)], verbose=False, early_stopping_rounds = 15)
        # eval model
        
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        mae = mean_absolute_error(y_test, predictions)
        if mae < temp_mae:
            # get feature that being important
            features = [feature for (feature,threshold) in feature_importances if threshold >= thresh]
            temp_mae = mae
            print("Optimal features: ", features)
        print("Thresh=%.3f, n=%d, MAE: %.3f" % (thresh, select_X_train.shape[1], mae))

if __name__ == "__main__":

    # ha noi  
    target = ['PM2.5']
    cols = constant.hanoi_features+target
    hanoi_dataset = pd.read_csv('data/csv/hanoi_data_median.csv', usecols=cols)
    feature_importances_xgboost(hanoi_dataset, cols)

