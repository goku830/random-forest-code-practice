# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np

# Load the Iris dataset from CSV file
data = pd.read_csv('iris.data', header=None, names=[
    "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
])

# Display the first few rows of the dataset (optional)
print("Dataset preview:")
print(data.head())

# Split the dataset into features (X) and labels (y)
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")

# Cross-validation to evaluate model generalization
scores = cross_val_score(clf, X, y, cv=5)
print(f"Cross-Validation Scores: {scores}")
print(f"Mean Cross-Validation Score: {scores.mean():.2f}")

# Feature importance analysis
importances = clf.feature_importances_
print("\nFeature Importance:")
for i, feature in enumerate(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']):
    print(f"{feature}: {importances[i]:.4f}")

# Scatter plot visualization using pandas data
# Define colors for each class
colors = {'Iris-setosa': 'blue', 'Iris-versicolor': 'green', 'Iris-virginica': 'red'}

# Create scatter plot for Sepal Length vs Sepal Width
plt.figure(figsize=(8, 6))
for class_label, color in colors.items():
    subset = data[data['class'] == class_label]
    plt.scatter(subset['sepal_length'], subset['sepal_width'],
                label=class_label, color=color, alpha=0.7)

# Add labels, legend, and title
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Sepal Width (Colored by Class)")
plt.legend(title="Classes", loc="lower right")
plt.grid(True)
plt.show()

# PCA Representation
# Map class names to numeric labels for PCA
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y_numeric = y.map(class_mapping)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA results
plt.figure(figsize=(8, 6))
for class_label, color in colors.items():
    subset = X_pca[y == class_label]
    plt.scatter(subset[:, 0], subset[:, 1], label=class_label, color=color, alpha=0.7)

# Add labels, legend, and title
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Representation of Iris Dataset")
plt.legend(title="Classes", loc="lower right")
plt.grid(True)
plt.show()

# Correlation Matrix Heatmap
# Compute the correlation matrix
correlation_matrix = X.corr()

# Visualize the correlation matrix with a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
