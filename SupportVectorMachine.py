import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import svm
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
training_accuracy = []

for kernel in kernels:
    clf = svm.SVC(kernel=kernel)

    clf.fit(x_train, y_train)
    prediction_test = clf.predict(x_test)
    prediction_training = clf.predict(x_train)
    test_accuracy.append(metrics.accuracy_score(y_test, prediction_test))
    training_accuracy.append(metrics.accuracy_score(y_train, prediction_training))

best_kernel = kernels[np.argmax(test_accuracy)]
print("The", best_kernel, "Kernel gives the best accuracy of", np.max(test_accuracy), "in the test set.")

best_kernel = kernels[np.argmax(training_accuracy)]
print("The", best_kernel, "Kernel gives the best accuracy of", np.max(training_accuracy), "in the training set.")

plt.plot(kernels, test_accuracy, marker = 'o', color = 'r',label='testing accuracy')
plt.plot(kernels, training_accuracy, marker = 'o', color = 'b',label='training accuracy')
plt.xlabel('Kernels')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Tuning hyperparameter 2: C
# 1. Using default kernel
c_values = np.arange(1,100)

test_accuracy = []
training_accuracy = []

for i in c_values:
    clf = svm.SVC(C=i)

    clf.fit(x_train, y_train)
    prediction_test = clf.predict(x_test)
    prediction_training = clf.predict(x_train)
    test_accuracy.append(metrics.accuracy_score(y_test, prediction_test))
    training_accuracy.append(metrics.accuracy_score(y_train, prediction_training))

plt.plot(c_values,test_accuracy, marker = 'o', color = 'r',label='testing accuracy')
plt.plot(c_values,training_accuracy, marker = 'o', color = 'b',label='training accuracy')
plt.xlabel('c_parameter')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("The highest accuracy of the default Kernel is", np.max(test_accuracy), "when C is", c_values[np.argmax(test_accuracy)], "in the training set")
print("The highest accuracy of the default Kernel is", np.max(training_accuracy), "when C is", c_values[np.argmax(training_accuracy)], "in the training set")

# 2. Using the best Kernel - linear
c_values = np.arange(1,100)

test_accuracy = []
training_accuracy = []

for i in c_values:
    clf = svm.SVC(kernel='linear', C=i)
    clf.fit(x_train, y_train)
    prediction_training = clf.predict(x_train)
    prediction_test = clf.predict(x_test)

    test_accuracy.append(metrics.accuracy_score(y_test, prediction_test))
    training_accuracy.append(metrics.accuracy_score(y_train, prediction_training))

plt.plot(c_values,test_accuracy, marker = 'o', color = 'r',label='testing accuracy')
plt.plot(c_values,training_accuracy, marker = 'o', color = 'b',label='training accuracy')
plt.xlabel('c_parameter')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

max_test_accuracy = np.max(test_accuracy)
max_test_idx = np.argmax(max_test_accuracy)

print("The highest accuracy of the linear kernel is", max_test_accuracy, "at C = ", max_test_idx, "in the test set")

max_training_accuracy = np.max(training_accuracy)
max_training_idx = np.argmax(max_training_accuracy)
print("The highest accuracy of the linear kernel is", max_training_accuracy, "at C = ", max_training_idx, "in the test set")
