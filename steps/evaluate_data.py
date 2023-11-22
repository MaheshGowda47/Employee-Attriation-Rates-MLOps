import logging 
import mlflow

import pandas as pd
from zenml import step
from zenml.client import Client

from src.evaluation import Classificationreport
from sklearn.base import ClassifierMixin
from typing import Dict
from typing_extensions import Annotated

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_df(model: ClassifierMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> Annotated[float, "classification_report"]:

    """
    evaluate the model on the ingested data
    """
    try:
        report_class = Classificationreport()
        y_pred = model.predict(X_test)
        report = report_class.calculate_score(y_test, y_pred)

        # Log individual metrics
        mlflow.log_metrics({
            "precision_0": report["0"]["precision"],
            "recall_0": report["0"]["recall"],
            "f1_score_0": report["0"]["f1-score"],
            "precision_1": report["1"]["precision"],
            "recall_1": report["1"]["recall"],
            "f1_score_1": report["1"]["f1-score"],
            "accuracy": report["accuracy"],
        })
        return report["accuracy"]
    except Exception as e:
        logging.error(f"error while evaluating model {e}")
        raise e