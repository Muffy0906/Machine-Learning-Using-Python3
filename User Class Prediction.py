#A program in Python to predict the class of user

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder

url = "~/Downloads/archive/heart.csv"

dataset = pd.read_csv(url)

array = dataset.values
X = array[:,:13]
Y = array[:,13]
Y = Y.astype('int')
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=10)

X_new = np.array([[46,0,2,142,177,0,0,160,1,1.4,0,0,2],[51,1,3,125,213,0,0,125,1,1.4,2,1,2]])

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print("model = LogisticRegression")
print("Accuracy: {}".format(accuracy_score(y_test, predictions)))
predictions = model.predict(X_new)
print("Prediction: {}\n".format(predictions))
 
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print("model = DecisionTreeClassifier")
print("Accuracy: {}".format(accuracy_score(y_test, predictions)))
predictions = model.predict(X_new)
print("Prediction: {}\n".format(predictions))
 
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)
print("model = RandomForestClassifier")
predictions = model.predict(x_test)
print("Accuracy: {}".format(accuracy_score(y_test, predictions)))
predictions = model.predict(X_new)
print("Prediction: {}".format(predictions))