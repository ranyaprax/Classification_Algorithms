import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# data sets
training_data = pd.read_csv('wildfires_training.csv')
test_data = pd.read_csv('wildfires_test.csv')

x_train = training_data.drop("fire", axis=1)
y_train = training_data['fire']
x_test = test_data.drop("fire", axis=1)
y_test = test_data['fire']

# default svm
clf = svm.SVC()
clf.fit(x_train,y_train)

prediction_svm = clf.predict(x_test)
print("Default SVM Accuracy:", metrics.accuracy_score(y_test, prediction_svm))

# Tuning Hyperparameter 1: Kernel
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

test_accuracy = []

for kernel in kernels:
    clf = svm.SVC(kernel=kernel)

    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    test_accuracy.append(metrics.accuracy_score(y_test, prediction))

best_kernel = kernels[np.argmax(test_accuracy)]
print("The", best_kernel, "Kernel gives the best accuracy of", np.max(test_accuracy))

plt.plot(kernels, test_accuracy, marker = 'o', color = 'r',label='testing accuracy')
plt.xlabel('Kernels')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Tuning hyperparameter 2: C
c_values = np.arange(1,100)

test_accuracy = []
for i in c_values:
    clf = svm.SVC(C=i)

    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    test_accuracy.append(metrics.accuracy_score(y_test, prediction))

plt.plot(c_values,test_accuracy, marker = 'o', color = 'r',label='testing accuracy')
plt.xlabel('c_parameter')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("The highest accuracy of the default Kernel is", np.max(test_accuracy), "when C is", c_values[np.argmax(test_accuracy)])


c_values = np.arange(1,100)

test_accuracy = []

for i in c_values:
    clf = svm.SVC(kernel='linear', C=i)
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    test_accuracy.append(metrics.accuracy_score(y_test, prediction))

plt.plot(c_values,test_accuracy, marker = 'o', color = 'r',label='testing accuracy')
plt.xlabel('c_parameter')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

max_test_accuracy = np.max(test_accuracy)
max_test_idx = np.argmax(max_test_accuracy)

print("The highest accuracy of the linear kernel is", max_test_accuracy, "at C = ", max_test_idx)
