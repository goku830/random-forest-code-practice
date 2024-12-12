import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Fetch the dataset
heart_disease = fetch_ucirepo(id=45)

# Access data and targets
X = pd.DataFrame(heart_disease.data.features, columns=heart_disease.data.feature_names)
y = pd.Series(heart_disease.data.targets.values.ravel(), name="target")

# Map target to binary classes: 0 (absence), 1 (presence)
y = y.apply(lambda x: 0 if x == 0 else 1)
# Example 2: Multi-level target (optional)
def severity_target(value):
    if value == 0:
        return 0  # No disease
    elif value in [1, 2]:
        return 1  # Low severity
    elif value in [3, 4]:
        return 2  # High severity y = y.apply(severity_target)

# Handle missing values in X
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Perform GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300,400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample']
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best estimator
best_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.5).astype(int)  # Lower threshold to prioritize recall


# Evaluate the model
print("Best Parameters:", grid_search.best_params_)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
