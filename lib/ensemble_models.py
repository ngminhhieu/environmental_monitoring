import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
sns.set(style='white', context='notebook', palette='deep')

def split_data(dataset, cols_feature, train_per, valid_per):
    # split data into X and y
    X = dataset.iloc[:,0:(len(cols_feature)-1)]
    Y = dataset.iloc[:,-1]
    # split data into train and test sets
    train_size = int(len(dataset)*train_per)
    valid_size = int(len(dataset)*valid_per)
    X_train = X.iloc[0:train_size]
    y_train = Y.iloc[0:train_size]

    X_valid = X.iloc[train_size:train_size+valid_size]
    y_valid = Y.iloc[train_size:train_size+valid_size]
    
    X_test = X.iloc[train_size:train_size+valid_size:]
    y_test = Y.iloc[train_size:train_size+valid_size:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def test_model(X_train, Y_train):
    # kfold = StratifiedKFold(n_splits=10)

    # Modeling step Test differents algorithms 
    random_state = 2
    regressors = []
    regressors.append(SVR())
    regressors.append(DecisionTreeRegressor(random_state=random_state))
    regressors.append(AdaBoostRegressor(DecisionTreeRegressor(random_state=random_state),random_state=random_state,learning_rate=0.1))
    regressors.append(RandomForestRegressor(random_state=random_state))
    regressors.append(ExtraTreesRegressor(random_state=random_state))
    regressors.append(GradientBoostingRegressor(random_state=random_state))
    regressors.append(XGBRegressor(random_state=random_state))
    regressors.append(MLPRegressor(random_state=random_state))
    regressors.append(KNeighborsRegressor())
    regressors.append(LogisticRegression(random_state = random_state))
    regressors.append(LinearDiscriminantAnalysis())
    regressors.append(make_pipeline(RobustScaler(), Lasso(random_state=random_state)))
    regressors.append(make_pipeline(RobustScaler(), ElasticNet(random_state=random_state)))
    regressors.append(KernelRidge())

    cv_results = []
    for regressor in regressors :
        print(regressor)
        cv_results.append(cross_val_score(regressor, X_train, y = Y_train, scoring = "neg_mean_absolute_error"))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
    "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis",
    "Lasso", "ElasticNet", "KernelRidge"]})

    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")

#Validation function
n_folds = 5

def mae_cv(model, X_train, Y_train):
    # kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    mae= cross_val_score(model, X_train, Y_train, scoring="neg_mean_absolute_error")
    return(mae)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)  

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

def mae(y, y_pred):
    return mean_absolute_error(y, y_pred)


if __name__ == "__main__":
    features = ['MONTH', 'DAY', 'YEAR', 'HOUR', 'AMB_TEMP', 'CO', 'NO', 'NO2',
    'NOx', 'O3', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR', 'PM10']    
    taiwan_dataset = pd.read_csv('data/csv/taiwan_data_mean.csv', usecols=features)
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(taiwan_dataset, features, 0.8, 0.2)
    test_model(X_train, y_train)

    # averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

    # score = rmsle_cv(averaged_models)   

    # stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
    #                                              meta_model = lasso)

    # score = rmsle_cv(stacked_averaged_models)
    # print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    # stacked_averaged_models.fit(train.values, y_train)
    # stacked_train_pred = stacked_averaged_models.predict(train.values)
    # stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
    # print(rmsle(y_train, stacked_train_pred))

    # model_xgb.fit(train, y_train)
    # xgb_train_pred = model_xgb.predict(train)
    # xgb_pred = np.expm1(model_xgb.predict(test))
    # print(rmsle(y_train, xgb_train_pred))

    # model_lgb.fit(train, y_train)
    # lgb_train_pred = model_lgb.predict(train)
    # lgb_pred = np.expm1(model_lgb.predict(test.values))
    # print(rmsle(y_train, lgb_train_pred))

    # '''RMSE on the entire Train data when averaging'''

    # print('RMSLE score on train data:')
    # print(rmsle(y_train,stacked_train_pred*0.70 +
    # xgb_train_pred*0.15 + lgb_train_pred*0.15 ))

    # ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15