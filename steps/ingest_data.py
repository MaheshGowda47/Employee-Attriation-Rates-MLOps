import logging

import pandas as pd
from zenml import step

"""
class to ingest data from source and return as dataframe

"""
class IngestData:
    def __init__(self, data_path):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"ingesting data from ingest data {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    try:
       ingest_data = IngestData(data_path)
       df = ingest_data.get_data()
       return df
    except Exception as e:
        logging.error(f"Error while ingesting data {e}")
        raise e