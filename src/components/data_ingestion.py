import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd

from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from dataclasses import dataclass #

@dataclass ## Decorator to automatically generate special methods like __init__() and __repr__() for the class
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    # These are the Input and Output file paths for the data ingestion process. 
    # The train_data_path and test_data_path are where the training and testing datasets will be saved, 
    # while raw_data_path is where the original dataset will be stored.

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # Create an instance of the DataIngestionConfig class and assign it to the ingestion_config attribute of the DataIngestion class.

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(os.path.join("notebooK\data\stud.csv")) 
            # Read the dataset from a specified path and store it in a DataFrame called df. 
            # It will read the CSV file located at "notebook/data/gemstone.csv", or any other path where the dataset is stored.
            logging.info("Read the dataset as dataframe")

            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) 
            # Create the directory for storing the training data if it doesn't already exist.

           
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) 
            # Save the original dataset to the raw_data_path specified in the configuration.

            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) 
            # Split the dataset into training and testing sets using an 80-20 split.

            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) 
            # Save the training set to the train_data_path specified in the configuration.

            
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) 
            # Save the testing set to the test_data_path specified in the configuration.

            
            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            ) # Return the file paths of the training and testing datasets.

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion() # Create an instance of the DataIngestion class.
    train_data,test_data = obj.initiate_data_ingestion() # Call the initiate_data_ingestion method to perform the data ingestion process and obtain the file paths of the training and testing datasets.    

    data_transformation = DataTransformation() 
     # Create an instance of the DataTransformation class.
    data_transformation.initiate_data_transformation(train_data, test_data) 
    # Call the initiate_data_transformation method to perform data
    #transformation on the training and testing datasets using the file paths obtained from the data ingestion process.   
