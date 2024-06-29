import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib


# Load the dataset with the correct delimiter
df = pd.read_csv('student/student-mat.csv', delimiter=';')

# Strip any whitespace from column names and convert to lowercase
df.columns = df.columns.str.strip().str.lower()

# Convert categorical variables to numeric
required_columns = ['schoolsup', 'famsup', 'paid', 'activities', 'higher', 'internet', 'romantic']
df = pd.get_dummies(df, columns=required_columns, drop_first=True)

# Define the features and target variable
target = 'g3'
features = ['age', 'absences', 'studytime', 'failures', 'schoolsup_yes', 'famsup_yes', 'paid_yes', 'activities_yes', 'higher_yes', 'internet_yes', 'romantic_yes', 'famrel', 'freetime', 'goout', 'dalc', 'walc', 'health', 'g1', 'g2']

# Ensure all feature columns are present
features = [col for col in features if col in df.columns]

# Define X and y
X = df[features]
y = df[target]

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42) # 60% training
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # 20% validation, 20% testing

# Save the datasets to CSV files
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Initialize and train a Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_rf_model = grid_search.best_estimator_

# Save the best model to a file
joblib.dump(best_rf_model, 'best_rf_model.pkl')


# Make predictions on training, validation and test sets
y_train_pred = best_rf_model.predict(X_train)
y_val_pred = best_rf_model.predict(X_val)
y_test_pred = best_rf_model.predict(X_test)

# Evaluate the model on training set
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
accuracy_train = r2_train * 100

# Evaluate the model on validation set
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)
accuracy_val = r2_val * 100

# Evaluate the model on test set
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
accuracy_test = r2_test * 100

# Print evaluation metrics
print(f'Training Mean Squared Error: {mse_train}')
print(f'Training R^2 Score: {r2_train}')
print(f'Training Mean Absolute Error: {mae_train}')
print(f'Training Accuracy: {accuracy_train:.2f}%')

print(f'Validation Mean Squared Error: {mse_val}')
print(f'Validation R^2 Score: {r2_val}')
print(f'Validation Mean Absolute Error: {mae_val}')
print(f'Validation Accuracy: {accuracy_val:.2f}%')

print(f'Test Mean Squared Error: {mse_test}')
print(f'Test R^2 Score: {r2_test}')
print(f'Test Mean Absolute Error: {mae_test}')
print(f'Test Accuracy: {accuracy_test:.2f}%')

# Example of predicting performance for a new student
new_student = np.array([[18, 3, 2, 0, 1, 1, 0, 1, 1, 0, 0, 4, 3, 3, 1, 1, 3, 14, 15]])
new_student_df = pd.DataFrame(new_student, columns=features)
prediction = best_rf_model.predict(new_student_df)
print(f'Predicted Final Grades (G3): new_student - {prediction[0]}')

# Plotting the performance
plt.figure(figsize=(14, 7))

# Plotting predicted vs actual values for validation set
plt.subplot(1, 2, 1)
plt.scatter(y_val, y_val_pred, alpha=0.7)
plt.xlabel('Actual Grades (Validation)')
plt.ylabel('Predicted Grades (Validation)')
plt.title('Actual vs Predicted Grades (Validation)')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', lw=2)
plt.grid()

# Plotting predicted vs actual values for test set
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.xlabel('Actual Grades (Test)')
plt.ylabel('Predicted Grades (Test)')
plt.title('Actual vs Predicted Grades (Test)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.grid()

plt.tight_layout()
plt.show()
