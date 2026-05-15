import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer ## To apply different transformations to different columns in the dataset
from sklearn.impute import SimpleImputer ## To handle missing values in the dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder ## To scale the features and encode categorical variables

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl") 
    # This is the file path where the preprocessor object will be saved after it is created. 
    # The preprocessor object contains the transformations that will be applied to the dataset during the data transformation process.  

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        # Create an instance of the DataTransformationConfig class and assign it to the data_transformation_config attribute of the DataTransformation class.

    def get_data_transformer_object(self):

        '''
        This function is responsible for creating and returning a preprocessor object that contains the transformations
        to be applied to the dataset. 
        It defines the numerical and categorical columns in the dataset, creates pipelines for both types of columns
        '''


        try:
            numerical_columns = ['writing_score', 'reading_score'] 
            # List of numerical columns in the dataset that will be transformed using a pipeline.

            categorical_columns = ['gender',
                                   'race_ethnicity',
                                   'parental_level_of_education', 
                                   'lunch', 
                                   'test_preparation_course'
                                ] 
            # List of categorical columns in the dataset that will be transformed using a pipeline.

            numerical_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')), 
                # We use Median strategy to impute missing values and there are some outliers in the numerical columns, 
                # as it is less affected by outliers compared to mean imputation.
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                # We use Most Frequent strategy to impute missing values in categorical columns, 
                # as it replaces missing values with the most common value in each column.
                ('one_hot_encoder', OneHotEncoder(drop='first')),
                ('scaler', StandardScaler(with_mean=False)) 
                # Optionally, you can scale the one-hot encoded features as well, 
                # but with_mean=False to avoid centering the sparse matrix.
            ])

            logging.info("Numerical columns encoding completed")
            logging.info(f"Numerical columns: {numerical_columns}")
            

            logging.info("Categorical columns encoding completed")
            logging.info(f"Categorical columns: {categorical_columns}")


            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical', numerical_pipeline, numerical_columns),
                    ('categorical', categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path) 
            # Read the training dataset from the specified file path and store it in a DataFrame called train_df.

            test_df = pd.read_csv(test_path) 
            # Read the testing dataset from the specified file path and store it in a DataFrame called test_df.

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformer_object() 
            # Call the get_data_transformer_object method to create and obtain the preprocessor object that contains the transformations to be applied to the dataset.

            target_column_name = 'math_score' 
            # Define the target column name that we want to predict.
            
            numerical_columns = ['writing_score', 'reading_score'] 
            # List of numerical columns in the dataset that will be transformed using a pipeline.

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1) 
            # Create a DataFrame called input_feature_train_df by dropping the target column from the training dataset. 
            # This DataFrame will contain only the input features for training.

            target_feature_train_df = train_df[target_column_name] 
            # Create a Series called target_feature_train_df that contains only the target column from the training dataset. 
            # This Series will be used as the target variable for training.

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1) 
            # Create a DataFrame called input_feature_test_df by dropping the target column from the testing dataset. 
            # This DataFrame will contain only the input features for testing.

            target_feature_test_df = test_df[target_column_name] 
            # Create a Series called target_feature_test_df that contains only the target column from the testing dataset. 
            # This Series will be used as the target variable for testing.

            logging.info("Applying preprocessing object on training and testing datasets.")

            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df) 
            # Apply the transformations defined in the preprocessor object to the input features of the training dataset and store the transformed features in an array called input_feature_train_arr.

            
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df) 
            # Apply the same transformations to the input features of the testing dataset using transform() method (without fitting again) and store the transformed features in an array called input_feature_test_arr.

            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] 
            # Combine the transformed input features and the target variable of the training dataset into a single array called train_arr.  

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] 
            # Combine the transformed input features and the target variable of the testing dataset into a single array called test_arr.    


            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            ) 
        # Return the transformed training and testing arrays, along with the file path of the preprocessor object.
        
        except Exception as e:
            raise CustomException(e, sys)