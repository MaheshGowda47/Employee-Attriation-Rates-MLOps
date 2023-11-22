import logging 
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.cleaning import DataCleaning, DataPreProcessingStrategy, DataDivideStrategy

@step
def clean_df(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_teat'],
]:
    
    """
    cleans the data and divides into train and test
    """
    try:
        process_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(data, process_strategy)
        processed_data = data_cleaning.handel_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handel_data()
        logging.info("Data cleaning completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in cleaning data {e}")
        raise e