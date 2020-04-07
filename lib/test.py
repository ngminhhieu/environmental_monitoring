from pandas import read_csv
import constant
from matplotlib import pyplot
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

train_per = 0.6
valid_per = 0.2

target_feature = ['PM2.5']
dataset_hanoi = 'data/csv/hanoi_data_median.csv'
dataset = read_csv(dataset_hanoi, usecols=constant.hanoi_features+target_feature)
corr = dataset.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

# pyplot.show()


# split data into X and y
X = dataset.iloc[:, 0:-1]
Y = dataset.iloc[:, -1]
# split data into train and test sets
train_size = int(len(dataset)*train_per)
valid_size = int(len(dataset)*valid_per)
X_train = X.iloc[0:train_size]
y_train = Y.iloc[0:train_size]

X_valid = X.iloc[train_size:train_size+valid_size]
y_valid = Y.iloc[train_size:train_size+valid_size]

X_test = X.iloc[train_size:train_size+valid_size:]
y_test = Y.iloc[train_size:train_size+valid_size:]



xgbregressor = XGBRegressor(objective ='reg:squarederror', max_depth=8, n_estimators=1000, min_child_weight=300, colsample_bytree=0.8, 
    subsample=0.8, eta=0.3, seed=2)

xgbregressor.fit(X_train.values, y_train.values, eval_metric="mae", eval_set=[(X_valid.values, y_valid.values)], verbose=False, early_stopping_rounds = 15)

# print(xgbregressor.feature_importances_)

# from sklearn.metrics import r2_score
# from rfpimp import permutation_importances

# def r2(rf, X_train, y_train):
#     return r2_score(y_train, rf.predict(X_train))

# X_train, X_valid, y_train, y_valid = train_test_split(dataset.iloc[:, 0:-1], dataset.iloc[:, -1], test_size = 0.8, random_state = 42)

# perm_imp_rfpimp = permutation_importances(xgbregressor, X_train, y_train, r2)
# print(perm_imp_rfpimp)

import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                   mode = 'regression',
                                                   feature_names = X_train.columns,
                                                   discretize_continuous = True)
                                                   
exp = explainer.explain_instance(X_train.values[31], xgbregressor.predict, num_features = 13)
exp.show_in_notebook(show_all=False)

print(exp)

from IPython.display import display, HTML
display(exp)

# exp = explainer.explain_instance(y_train.to_numpy(), xgbregressor.predict, num_features = 13)
# exp.show_in_notebook(show_all=False)