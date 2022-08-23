import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import joblib

features = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race" , "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]


data_train = pd.read_csv("adult.data", sep=" ", header=None)
data_train.columns = features
X_train = data_train[features[:-1]]
y_train = data_train["income"]
print(X_train.shape, y_train.shape)


data_test = pd.read_csv("adult.test", sep=",", header=None)
data_test.columns = features
X_test = data_test[features[:-1]]
y_test = data_test["income"]
print(X_test.shape, y_test.shape)

train_mode = dict(X_train.mode().iloc[0]) # mode() để lấy giá trị xuất hiện nhiều và iloc để xác định vị trí
X_train = X_train.fillna(train_mode) # fill giá trị NaN, Na, Null, ... bằng param đưa vào

features_categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex','native-country']
encoders = {}
for column in features_categorical:
    X_train[column] = LabelEncoder().fit_transform(X_train[column])
    encoders[column] = LabelEncoder()

for c in features[:-1]:
    if X_train[c].dtypes != "int64":
        X_train[c] = X_train[c].apply(lambda x: x.split(',')[0]).astype('int')

# train the Random Forest algorithm
rf = RandomForestClassifier(n_estimators = 100) # n_estimators: số  lượng trees
rf = rf.fit(X_train, y_train)

# train the Extra Trees algorithm
et = ExtraTreesClassifier(n_estimators = 100)
et = et.fit(X_train, y_train)

# save preprocessing objects and weights
joblib.dump(train_mode, "./train_mode.joblib", compress=True)
joblib.dump(encoders, "./encoders.joblib", compress=True)
joblib.dump(rf, "./random_forest.joblib", compress=True)
joblib.dump(et, "./extra_trees.joblib", compress=True)



