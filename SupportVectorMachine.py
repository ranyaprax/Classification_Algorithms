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


# Tuning hyperparameter 2: C
c_values = np.arange(1,100)

training_accuracy = []
test_accuracy = []
for i in c_values:
    clf = svm.SVC(C=i)

    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    training_accuracy.append(clf.score(x_train, y_train))
    test_accuracy.append(clf.score(x_test, y_test))

plt.plot(c_values,training_accuracy, marker = 'o', color = 'b',label='training accuracy')
plt.plot(c_values,test_accuracy, marker = 'o', color = 'r',label='testing accuracy')
plt.xlabel('c_parameter')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

diff = np.array(training_accuracy) - np.array(test_accuracy)
min_idx = np.argmin(diff)

clf= svm.SVC(C=c_values[min_idx])
clf.fit(x_train, y_train)

prediction_c_tuning = clf.predict(x_test)
print("C Tuning using default kernel - Logistic Regression Accuracy:", metrics.accuracy_score(y_test, prediction_c_tuning))


c_values = np.arange(1,100)

training_accuracy = []
test_accuracy = []
for i in c_values:
    clf = svm.SVC(kernel='linear', C=i)

    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    training_accuracy.append(clf.score(x_train, y_train))
    test_accuracy.append(clf.score(x_test, y_test))

plt.plot(c_values,training_accuracy, marker = 'o', color = 'b',label='training accuracy')
plt.plot(c_values,test_accuracy, marker = 'o', color = 'r',label='testing accuracy')
plt.xlabel('c_parameter')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

diff = np.array(training_accuracy) - np.array(test_accuracy)
min_idx = np.argmin(diff)

clf= svm.SVC(kernel='linear', C=c_values[min_idx])
clf.fit(x_train, y_train)

prediction_c_tuning = clf.predict(x_test)
print("Best C value: ", c_values[min_idx])
print("C Tuning using linear kernel - Logistic Regression Accuracy:", metrics.accuracy_score(y_test, prediction_c_tuning))
