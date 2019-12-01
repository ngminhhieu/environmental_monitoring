import seaborn as sea
import pandas as pd
import numpy as np

titanic = sea.load_dataset("titanic")
titanic = titanic.drop(columns = ['deck'])

from sklearn.ensemble import RandomForestRegressor

titanicWithAge = titanic[pd.isnull(titanic['age']) == False]
# print(titanicWithAge)
titanicWithoutAge = titanic[pd.isnull(titanic['age'])]
print(titanicWithoutAge)
variables = ['pclass', 'sibsp', 'parch', 'fare', 'age']
one_hot_encoded_embarked = pd.get_dummies(titanicWithAge['embarked'])
one_hot_encoded_sex = pd.get_dummies(titanicWithAge['sex'])
titanicWithAge = titanicWithAge[variables]
titanicWithAge = pd.concat([titanicWithAge, one_hot_encoded_sex, one_hot_encoded_embarked], axis = 1)

one_hot_encoded_embarked = pd.get_dummies(titanicWithoutAge['embarked'])
one_hot_encoded_sex = pd.get_dummies(titanicWithoutAge['sex'])
titanicWithoutAge = titanicWithoutAge[variables]
titanicWithoutAge = pd.concat([titanicWithoutAge, one_hot_encoded_sex, one_hot_encoded_embarked], axis = 1)

independentVariables = ['pclass', 'female', 'male', 'sibsp', 'parch', 'fare', 'C', 'Q', 'S']

rfModel_age = RandomForestRegressor()
rfModel_age.fit(titanicWithAge[independentVariables], titanicWithAge['age'])

generatedAgeValues = rfModel_age.predict(X = titanicWithoutAge[independentVariables])
print(generatedAgeValues)
titanicWithoutAge['age'] = generatedAgeValues.astype(int)
data = titanicWithAge.append(titanicWithoutAge)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)