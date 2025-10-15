from turtledemo.clock import current_day

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
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

# Tuning hyperparameter 2: Solver + Penalty

# a parameter grid with compatible combinations
param_grid = [
    {'penalty': ['l1'], 'solver': ['liblinear', 'saga']},
    {'penalty': ['l2'], 'solver': ['lbfgs', 'newton-cg', 'sag', 'saga']},
    {'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0.3, 0.5, 0.7]}
]

best_score = 0
all_scores = []
best_params = []

# try all combination of penalty and solvers
for params in param_grid:
    penalties = params.get('penalty', ['l2'])
    solvers = params.get('solver', ['lbfgs'])
    l1_ratios = params.get('l1_ratio', [None]) # for elasticnet only

    for penalty in penalties:
        for solver in solvers:
            for l1_ratio in l1_ratios:
                try:
                    if penalty == 'elasticnet':
                        model = LogisticRegression(penalty=penalty, solver=solver, max_iter=370, l1_ratio=l1_ratio)
                        label = f"{penalty}-:{solver}-l1_ratio:{l1_ratio}"
                    else:
                        model = LogisticRegression(penalty=penalty, solver=solver, max_iter=370)
                        label = f"{penalty}-:{solver}"

                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    score = accuracy_score(y_test, y_pred)

                    all_scores.append({
                        'Combination': label,
                        'Accuracy': score
                    })

                    if score > best_score:
                        best_score = score
                        best_params = {'penalty': penalty, 'solver': solver}

                except Exception:
                    continue

print("Best validation accuracy:", best_score)
print("Best parameters:", best_params)

df_results = pd.DataFrame(all_scores)

# Plot accuracy for each combination of penalty/solver pair
plt.figure(figsize=(10, 5))
plt.plot(df_results['Combination'], df_results['Accuracy'], color='darkblue')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Penalty and Solver Combination')
plt.ylabel('Accuracy')
plt.title('Accuracy for Each Penalty and Solver Combination')
plt.tight_layout()
plt.show()

