import logging 
import numpy as np
from sklearn.metrics import classification_report
from abc import ABC, abstractmethod

"""
Abstract class for all model
"""
class Evaluation(ABC):
    """
    class to define the strategy and evaluate the performance of the model
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class Classificationreport(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("calculate classification report")
            report = classification_report(y_true, y_pred, output_dict=True)
            logging.info(f"Classification Report : \n{report}")
            return report
        except Exception as e:
            logging.error(f"Error while calculating classification report {e}")
            raise e