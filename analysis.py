import pandas as pd
from sklearn import svm
from sklearn import metrics

# This file is for analysis and research of a few algorithms before choosing

training_data = pd.read_csv('wildfires_training.csv')
test_data = pd.read_csv('wildfires_test.csv')

x_train = training_data.drop("fire", axis=1)
y_train = training_data['fire']
x_test = test_data.drop("fire", axis=1)
y_test = test_data['fire']

# Support Vector Machine algorithm

clf = svm.SVC()
clf.fit(x_train,y_train)

prediction_svm = clf.predict(x_test)
print("SVM Accuracy:", metrics.accuracy_score(y_test, prediction_svm))


# Random Forests
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

prediction_rf = rf.predict(x_test)
print("Random Forest Accuracy:", metrics.accuracy_score(y_test, prediction_rf))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)

prediction_nb = nb.predict(x_test)
print("Naive Bayes Accuracy:", metrics.accuracy_score(y_test, prediction_nb))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

prediction_lr = lr.predict(x_test)
print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, prediction_lr))

# Neural Network
from sklearn.neural_network import MLPClassifier

model = MLPClassifier()
model.fit(x_train, y_train)

prediction_model = model.predict(x_test)
print("MLP Model Accuracy:", metrics.accuracy_score(y_test, prediction_model))