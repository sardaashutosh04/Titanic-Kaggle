# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")

def is_alone(x):
    if  (x['SibSp'] + x['Parch'])  > 0:
        return 1
    else:
        return 0

train['Is_alone'] = train.apply(is_alone, axis = 1)
test['Is_alone'] = test.apply(is_alone, axis = 1)

train = train.drop(['PassengerId','Name','SibSp','Parch'], axis = 1)
test = test.drop(['Name','SibSp','Parch'], axis = 1)

numerical = ['Pclass', 'Age', 'Fare', 'Is_alone']
categorical = ['Sex', 'Ticket', 'Cabin', 'Embarked']

features = numerical + categorical
target = ['Survived']

from sklearn.model_selection import train_test_split
train_set, valid_set = train_test_split(train, test_size = 0.3, random_state = 0)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
numerical_transformer = Pipeline(steps=[
                        ('simple', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
               transformers=[
                    ('num', numerical_transformer, numerical),
                    ('cat', categorical_transformer, categorical)])

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=0.1,random_state = 0)
pipe = Pipeline(steps = [('preprocessor', preprocessor), 
                         ('model', model)])
pipe.fit(train_set[features], np.ravel(train_set[target]))
pred_valid = pipe.predict(valid_set[features])

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

acc_ran_train = round(pipe.score(train_set[features], train_set[target]) * 100, 2)
acc_ran_valid = round(pipe.score(valid_set[features], valid_set[target]) * 100, 2)

pred_test = pipe.predict(test[features])
output= pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':pred_test})

output.to_csv('subission.csv', index=False)





