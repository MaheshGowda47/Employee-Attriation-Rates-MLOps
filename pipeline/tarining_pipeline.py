import pandas as pd
from zenml import pipeline

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.evaluate_data import evaluate_df
from steps.model_data import train_df

@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_df(X_train, X_test, y_train, y_test)
    report = evaluate_df(model,X_test, y_test)
