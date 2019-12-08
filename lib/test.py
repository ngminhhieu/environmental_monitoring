from sklearn.ensemble import RandomForestClassifier
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn import utils
df_train = pd.read_csv("data/predicted_data.csv")

df_train['time'] = pd.to_datetime(df_train['time'])
df_train['time'] = pd.to_timedelta(df_train['time'])
df_train['time'] = df_train['time'] / pd.offsets.Minute(1)

model = RandomForestClassifier()
training_score_y = df_train['pm_2.5'].copy()
training_input = df_train.drop(['pm_2.5'], axis=1).copy()

# convert continuous to multiclass for training_score_y
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(training_score_y)
print(utils.multiclass.type_of_target(training_input))
print(utils.multiclass.type_of_target(training_score_y))
print(utils.multiclass.type_of_target(training_scores_encoded))

model.fit(training_input, training_scores_encoded)

importances = model.feature_importances_
#Sort it
print ("Sorted Feature Importance:")
sorted_feature_importance = sorted(zip(importances, list(training_input)), reverse=True)
print (sorted_feature_importance)