import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


training_data = pd.read_csv('wildfires_training.csv')
test_data = pd.read_csv('wildfires_test.csv')

x_train = training_data.drop("fire", axis=1)
y_train = training_data['fire']
x_test = test_data.drop("fire", axis=1)
y_test = test_data['fire']

clf = svm.SVC()
clf.fit(x_train,y_train)

prediction_svm = clf.predict(x_test)
print("Default SVM Accuracy:", metrics.accuracy_score(y_test, prediction_svm))

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

training_accuracy = []
test_accuracy = []

for kernel in kernels:
    clf = svm.SVC(kernel=kernel)

    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    training_accuracy.append(clf.score(x_train, y_train))
    test_accuracy.append(clf.score(x_test, y_test))

print("Best kernel to use: Linear - ", test_accuracy[0])

plt.plot(kernels, training_accuracy, marker = 'o', color = 'b',label='training accuracy')
plt.plot(kernels, test_accuracy, marker = 'o', color = 'r',label='testing accuracy')
plt.xlabel('Kernels')
plt.ylabel('Accuracy')
plt.legend()
plt.show()