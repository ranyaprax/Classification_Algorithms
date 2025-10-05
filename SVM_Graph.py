from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

training_data = pd.read_csv('wildfires_training.csv')

X = training_data.drop("fire", axis=1)
y = training_data["fire"]

# use Random Forest to get importance of each attribute
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Get feature importance list
importance_list = pd.Series(rf.feature_importances_, index=X.columns)
importance_list = importance_list.sort_values(ascending=False)

print(importance_list)

# Pick the top 2 features to use for the graph
top_features = importance_list.index[:2]
print("Top 2 features:", top_features)

# Plot using the top 2 features
plt.figure(figsize=(8,6))
y = y.map({"no": 0, "yes": 1})
plt.scatter(training_data[top_features[0]], training_data[top_features[1]], c=y, cmap='viridis', edgecolors='k')
plt.xlabel(top_features[0])
plt.ylabel(top_features[1])
plt.title("Training Data - SVM")
plt.show()