import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import base64
from io import BytesIO
from flask import Flask, request, render_template
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('best_rf_model.pkl')

# Load the datasets
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()  # Ensure y_train is a 1D array
X_val = pd.read_csv('X_val.csv')
y_val = pd.read_csv('y_val.csv').values.ravel()  # Ensure y_val is a 1D array
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()  # Ensure y_test is a 1D array

# Define the features
features = ['age', 'absences', 'studytime', 'failures', 'schoolsup_yes', 'famsup_yes', 'paid_yes', 'activities_yes',
            'higher_yes', 'internet_yes', 'romantic_yes', 'famrel', 'freetime', 'goout', 'dalc', 'walc', 'health', 'g1', 'g2']

# Function to plot actual vs predicted values
def plot_actual_vs_predicted(y_actual, y_predicted, title):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_actual, y_predicted, alpha=0.7)
    plt.xlabel('Actual Grades')
    plt.ylabel('Predicted Grades')
    plt.title(title)
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red', lw=2)
    plt.grid()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

    return graph_url

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_features = [request.form.get(feature) for feature in features]
    input_features = np.array(input_features, dtype=np.float64).reshape(1, -1)
    input_df = pd.DataFrame(input_features, columns=features)

    prediction = model.predict(input_df)[0]

    # Make predictions on training, validation, and test sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    accuracy_train = r2_train * 100

    mse_val = mean_squared_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    accuracy_val = r2_val * 100

    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    accuracy_test = r2_test * 100

    # Generate graphs
    graph_train = plot_actual_vs_predicted(y_train, y_train_pred, 'Actual vs Predicted Grades (Training Set)')
    graph_val = plot_actual_vs_predicted(y_val, y_val_pred, 'Actual vs Predicted Grades (Validation Set)')
    graph_test = plot_actual_vs_predicted(y_test, y_test_pred, 'Actual vs Predicted Grades (Test Set)')

    return render_template('index.html', prediction=prediction, mse_train=mse_train, r2_train=r2_train,
                           mae_train=mae_train, accuracy_train=accuracy_train, mse_validation=mse_val,
                           r2_validation=r2_val, mae_validation=mae_val, accuracy_validation=accuracy_val,
                           mse_test=mse_test, r2_test=r2_test, mae_test=mae_test, accuracy_test=accuracy_test,
                           graph_train=graph_train, graph_validation=graph_val, graph_test=graph_test)

if __name__ == '__main__':
    app.run(debug=True)
