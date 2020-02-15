import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
# import utils
import utils

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

if __name__ == "__main__":
    
    # get dataset
    features = ['MONTH', 'DAY', 'YEAR', 'HOUR', 'AMB_TEMP', 'CO', 'NO', 'NO2',
    'NOx', 'O3', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR', 'PM10']
    # change later
    print("--Split data--")    
    taiwan_dataset = pd.read_csv('data/csv/taiwan_test.csv', usecols=features)
    X_train, y_train, X_valid, y_valid, X_test, y_test = utils.split_data(taiwan_dataset, features, 0.8, 0.2)

    # get models
    print("--Starting get models--")
    models = utils.get_models("Lasso", "ElasticNet", "KernelRidge", "GradientBoostingRegressor", "LGBMRegressor", "XGBRegressor")

    ENet = models["ElasticNet"]
    GBoost = models["GradientBoostingRegressor"]
    KRR = models["KernelRidge"]
    lasso = models["Lasso"]
    model_xgb = models["XGBRegressor"]
    models_lgb = models["LGBMRegressor"]
    print("--Done get models!--")

    ### Stacking 
    # Avergage models
    averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

    score = utils.mae_cv(averaged_models, X_train, y_train) 
    print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    # Stacked average models
    stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                    meta_model = lasso)
    score = utils.mae_cv(stacked_averaged_models, X_train, y_train)
    print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


    stacked_averaged_models.fit(X_train, y_train)
    stacked_train_pred = stacked_averaged_models.predict(X_train)
    stacked_pred = np.expm1(stacked_averaged_models.predict(X_test))
    print(utils.mae(y_train, stacked_train_pred))

    model_xgb.fit(X_train, y_train)
    xgb_train_pred = model_xgb.predict(X_train)
    xgb_pred = np.expm1(model_xgb.predict(X_test))
    print(utils.mae(y_test, xgb_pred))

    model_lgb.fit(X_train, y_train)
    lgb_train_pred = model_lgb.predict(X_train)
    lgb_pred = np.expm1(model_lgb.predict(X_test))
    print(utils.mae(y_test, lgb_pred))

    '''MAE on the entire Train data when averaging'''

    print('MAE score on train data:')
    print(utils.mae(y_train,stacked_train_pred*0.70 +
    xgb_train_pred*0.15 + lgb_train_pred*0.15 ))

    ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15