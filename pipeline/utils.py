import logging
import pandas as pd
from src.cleaning import DataCleaning, DataPreProcessingStrategy

# Updated get_data_for_test() function
def get_data_for_test():
    try:
        df = pd.read_csv("./data/HR-Employee-Attrition.csv")
        #df = df.sample(n=100)
        print("Original Data sample:")
        print(df.head())
        # Create a DataPreProcessStrategy instance with encoder
        preprocess_strategy = DataPreProcessingStrategy() 
        # Data cleaning with preprocessing
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handel_data()
        print("Preprocessed Data:")
        print(df.head())
        # Drop 'Attrition' column from test data
        df.drop(["Attrition"], axis=1, inplace=True)
        print("Data Shape for Inference:", df.shape)  # Add this line to print the shape
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e
