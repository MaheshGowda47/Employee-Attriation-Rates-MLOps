import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression


"""
Abstract class for all model
"""
class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        """
        pass

class LogisticRegressionModel(Model):
    """
    Logistic regression model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        train the model
        """
        try:
            model = LogisticRegression(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training complete")
            return model
        except Exception as e:
            logging.error(f"error in training model {e}")
            raise e