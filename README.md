This is a Student Performance Prediction Model which uses the random forest algorithom.
This project involves developing a predictive model to estimate the final grades (G3) of students based on various factors such as their demographics, study time, and past academic performance. The dataset used is the "Student Performance" dataset, which includes information about students in a Portuguese secondary school. The model employs a Random Forest Regressor, which is a robust ensemble learning method known for its accuracy and stability.
Model Description

The Random Forest Regressor model is trained on the provided student data to predict their final grades. Key steps include:

    Preprocessing the data by converting categorical variables to numeric and handling any missing values.
    Splitting the data into training, validation, and test sets.
    Performing hyperparameter tuning using GridSearchCV to find the best parameters for the model.
    Evaluating the model's performance using metrics like Mean Squared Error (MSE), R-squared score (R2), and Mean Absolute Error (MAE).

Technologies Used

    Python: The primary programming language used for model development.
    Pandas & NumPy: For data manipulation and numerical computations.
    Scikit-learn: For machine learning model training and evaluation.
    Joblib: For model serialization and saving.
    Matplotlib: For plotting and visualizing the results.
    Flask: For building the web application interface.
    Jinja2: For rendering HTML templates in the Flask application.

How to Run this Model

   Run the newRandomForest.py script to preprocess the data, train the Random Forest model.
   Start the Flask web application to interact with the model by running the app.py
   Open your web browser and navigate to http://127.0.0.1:5000 to use the web interface for predicting student grades.

After running the Flask application, you can enter details about a new student (such as age, study time, past grades, etc.) into the web form. The model will predict the final grade (G3) based on the input features and display the result.
