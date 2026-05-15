# It contains common utility functions that can be used across the project.
import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) 
        # Get the directory path from the file path where the object will be saved.

        os.makedirs(dir_path, exist_ok=True) 
        # Create the directory if it doesn't already exist.

        with open(file_path, "wb") as file_obj: 
            # Open the file in binary write mode and save the object using dill.
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i] 
            # Get the model object from the models dictionary using its index.

            model.fit(X_train, y_train) 
            # Fit the model on the training data.

            y_train_pred = model.predict(X_train) 
            # Make predictions on the training data using the fitted model.

            y_test_pred = model.predict(X_test) 
            # Make predictions on the testing data using the fitted model.

            train_model_score = r2_score(y_train, y_train_pred) 
            # Calculate the R-squared score of the model's predictions on the training data.

            test_model_score = r2_score(y_test, y_test_pred) 
            # Calculate the R-squared score of the model's predictions on the testing data.

            report[list(models.keys())[i]] =  test_model_score
            # Store the R-squared score in a report dictionary with the model name as the key.

        return report

    except Exception as e:
        raise CustomException(e, sys)
    