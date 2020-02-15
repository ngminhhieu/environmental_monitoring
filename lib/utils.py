import pandas as pd

from sklearn.metrics import mean_absolute_error

# models
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

# support for models
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

# plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

error_invalid_model = "Invalid Models"

def mae(y, y_pred):
    return mean_absolute_error(y, y_pred)

# Cross validation
def mae_cv(model, X_train, Y_train, n_folds = 10):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    mae= -cross_val_score(model, X_train, Y_train, scoring="neg_mean_absolute_error", cv=kf)
    return(mae)

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

def test_models(X_train, Y_train):
    # Modeling step Test differents algorithms 
    random_state = 42
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
    # regressors.append(LogisticRegression(random_state = random_state))
    # regressors.append(LinearDiscriminantAnalysis())
    regressors.append(make_pipeline(RobustScaler(), Lasso(random_state=random_state)))
    regressors.append(make_pipeline(RobustScaler(), ElasticNet(random_state=random_state)))
    regressors.append(KernelRidge())

    cv_results = []
    for regressor in regressors :
        # print(regressor)
        cv_results.append(mae_cv(regressor, X_train, Y_train))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())
    
    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVR","DecisionTree","AdaBoost",
    "RandomForest","ExtraTrees","GradientBoosting", "XGBoost", "MultipleLayerPerceptron","KNeighboors", "Lasso", "ElasticNet", "KernelRidge"]})
    # "LogisticRegression",
    # "LinearDiscriminantAnalysis"
    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    plt.show()

def switch_model(model):
    lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

    GBRegressor = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

    switcher={
        "SVR":'Sunday',
        "DecisionTreeRegressor":'Monday',
        "AdaBoostRegressor":'Tuesday',
        "ExtraTreesRegressor":'Wednesday',
        "GradientBoostingRegressor": GBRegressor,
        "XGBRegressor":'Friday',
        "MLPRegressor":'Saturday',
        "KNeighborsRegressor": "...",
        "Lasso": lasso,
        "ElasticNet": ENet,
        "KernelRidge": KRR
    }

    return switcher.get(model, error_invalid_model)

def get_models(*args):
    models = {}
    for model_name in args:
        model = switch_model(model_name)
        if model == error_invalid_model:
            continue
        else:
            models[model_name] = model
    
    return models
