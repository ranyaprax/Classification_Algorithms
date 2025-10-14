import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
import warnings

training_data = pd.read_csv('wildfires_training.csv')
test_data = pd.read_csv('wildfires_test.csv')

x_train = training_data.drop("fire", axis=1)
y_train = training_data['fire']
x_test = test_data.drop("fire", axis=1)
y_test = test_data['fire']

# Default
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

prediction_default = lr.predict(x_test)
print("Default Logistic Regression Accuracy:", metrics.accuracy_score(y_test, prediction_default))

cm = confusion_matrix(prediction_default, y_test)

sns.heatmap(cm,
            annot=True,
            fmt='g',
            cmap='Blues',
            xticklabels=['Yes', 'No'],
            yticklabels=['Yes', 'No'],
            square=True,
            cbar=False)
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Default LR - Fire Prediction vs Actual', fontsize=17)
plt.show()

# Tuning hyperparameter 1: C
c_values = np.arange(1,50)

training_accuracy = []
test_accuracy = []
for i in c_values:
    lr = LogisticRegression(max_iter=5000, C=i)

    lr.fit(x_train, y_train)
    prediction = lr.predict(x_test)
    training_accuracy.append(lr.score(x_train, y_train))
    test_accuracy.append(lr.score(x_test, y_test))

plt.plot(c_values,training_accuracy, marker = 'o', color = 'b',label='training accuracy')
plt.plot(c_values,test_accuracy, marker = 'o', color = 'r',label='testing accuracy')
plt.xlabel('c_parameter')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

diff = np.array(training_accuracy) - np.array(test_accuracy)
min_idx = np.argmin(diff)

lr = LogisticRegression(max_iter=1000, C=c_values[min_idx])
lr.fit(x_train, y_train)

prediction_c_tuning = lr.predict(x_test)
print("C Tuning - Logistic Regression Accuracy:", metrics.accuracy_score(y_test, prediction_c_tuning))

cm_c_tuning = confusion_matrix(prediction_c_tuning, y_test)

sns.heatmap(cm_c_tuning,
            annot=True,
            fmt='g',
            cmap='Blues',
            xticklabels=['Yes', 'No'],
            yticklabels=['Yes', 'No'],
            square=True,
            cbar=False)
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('C Tuning LR - Fire Prediction vs Actual', fontsize=17)
plt.show()

# finding best max_iter for convergence
max_iter = 100
for max_iter in range(100, 1000, 10):
    clf = LogisticRegression(max_iter=max_iter)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x_train, y_train)
        if not any(isinstance(wi.message, ConvergenceWarning) for wi in w):
            print(f"Converged with max_iter={max_iter}")
            break
