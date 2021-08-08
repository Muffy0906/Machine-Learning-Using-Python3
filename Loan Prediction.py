#A program in Python to predict if a loan will get approved or not.

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

url = "https://raw.githubusercontent.com/callxpert/datasets/master/Loan-applicant-details.csv"
names = ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
dataset = pd.read_csv(url, names=names)

var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    dataset[i] = le.fit_transform(dataset[i])
array = dataset.values
X = array[:,4:11]
Y = array[:,12]
Y=Y.astype('int')
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=10)

X_new = np.array([[0,0,4583,1508,128,360,1],[0,1,3000,0,66,360,1]])

model = LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print("Accuracy: {}".format(accuracy_score(y_test, predictions)))
predictions = model.predict(X_new)
print("Prediction: {}".format(predictions))
 
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print("Accuracy: {}".format(accuracy_score(y_test, predictions)))
predictions = model.predict(X_new)
print("Prediction: {}".format(predictions))
 
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print("Accuracy: {}".format(accuracy_score(y_test, predictions)))
predictions = model.predict(X_new)
print("Prediction: {}".format(predictions))

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print("Accuracy: {}".format(accuracy_score(y_test, predictions)))
predictions = model.predict(X_new)
print("Prediction: {}".format(predictions))