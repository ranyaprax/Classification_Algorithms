import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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


