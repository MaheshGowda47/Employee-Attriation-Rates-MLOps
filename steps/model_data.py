import logging

import pandas as pd
import mlflow
from zenml import step
from zenml.client import Client

from src.model import LogisticRegressionModel
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig

experiment_tracker=Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_df(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> ClassifierMixin:
    """
    Trains the model on the ingested data
    """
    try:
        model = None
        if config.model_name == "LogisticRegression":
            mlflow.sklearn.autolog()
            model = LogisticRegressionModel()
            trainde_model = model.train(X_train, y_train)
            return trainde_model
        else:
            raise ValueError(f"model {config.model_name} not supported")
    except Exception as e:
        logging.error(f"error while training the model {e}")
        raise e