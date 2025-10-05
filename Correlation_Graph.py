import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

training_data = pd.read_csv('wildfires_training.csv')
training_data["fire"] = training_data["fire"].map({'no': 0, 'yes': 1})

correlation = training_data.corr()  # computes pair correlations

target_corr = correlation['fire'].sort_values(ascending=False)
print(target_corr)

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()