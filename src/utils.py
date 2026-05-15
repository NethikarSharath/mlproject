# It contains common utility functions that can be used across the project.
import os
import sys
import dill
import numpy as np
import pandas as pd


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
    