import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_excel('iris.xls')

x = data.drop(['Classification'], axis = 1)
y = data['Classification']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25, random_state = 1)

logit_model = Logistic Regression()
# Fit this into xtrain and ytrain to create the model
logit_model.fit(x_train, y_train)


pickle.dump(logit_model,open('model.pickle', 'wb') )