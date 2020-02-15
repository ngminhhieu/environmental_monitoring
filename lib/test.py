from sklearn.base import TransformerMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd

class RidgeTransformer(Ridge, TransformerMixin):
    
    def transform(self, X, *_):
        return self.predict(X)


class RandomForestTransformer(RandomForestRegressor, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X)


class KNeighborsTransformer(KNeighborsRegressor, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X)

def build_model():
    ridge_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly_feats', PolynomialFeatures()),
        ('ridge', RidgeTransformer())
    ])

    pred_union = FeatureUnion(
        transformer_list=[
            ('ridge', ridge_transformer),
            ('rand_forest', RandomForestTransformer()),
            ('knn', KNeighborsTransformer())
        ],
        n_jobs=2
    )

    model = Pipeline(steps=[
        ('pred_union', pred_union),
        ('lin_regr', LinearRegression())
    ])

    return model

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

print('Build and fit a model...')

model = build_model()

features = ['MONTH', 'DAY', 'YEAR', 'HOUR', 'AMB_TEMP', 'CO', 'NO', 'NO2',
    'NOx', 'O3', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR', 'PM10']    
taiwan_dataset = pd.read_csv('data/csv/taiwan_data_mean.csv', usecols=features)
X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(taiwan_dataset, features, 0.8, 0.2)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print('Done. Score:', score)