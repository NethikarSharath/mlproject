import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor,
                              AdaBoostRegressor,
                              GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor    

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl") 
    # This is the file path where the trained model will be saved after it is created. 
        # The trained model file will contain the parameters and structure of the model that has been trained on the dataset, and it can be used for making predictions on new data in the future.  

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 
        # Create an instance of the ModelTrainerConfig class and assign it to the model_trainer_config attribute of the ModelTrainer class.

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            ) 
            # Split the training and testing arrays into input features (X) and target variable (y) for both training and testing datasets. 
                # The input features are obtained by selecting all columns except the last one (:-1), while the target variable is obtained by selecting only the last column (-1). 
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                #"R2 Score": r2_score(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }


            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values())) 
            ## to get the best score from the model report dictionary which contains the R-squared scores of all the models 
                # evaluated on the testing dataset. 
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
             ## to get the name of the best model corresponding to the best score from the model report dictionary.

            best_model = models[best_model_name] 
            # to get the best model object from the models dictionary using the name of the best model obtained from the model report. 

            if best_model_score < 0.6:
                raise CustomException("No best model found") 
                # If the best model's R-squared score is less than 0.6, raise a CustomException indicating that no suitable model was found.    
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted= best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square
            # Return the R-squared score of the best model's predictions on the testing dataset, 
                # which indicates how well the model is performing in terms of explaining the variance in the target 
                    # variable based on the input features.   

        except Exception as e:
            raise CustomException(e, sys)