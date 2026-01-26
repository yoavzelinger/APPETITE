import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier

class PriorDataDecisionTreeClassifierWrapper(DecisionTreeClassifier):
    """
    A wrapper for DecisionTreeClassifier that accepts prior data for training.
    """
    def __init__(self,
                 model: DecisionTreeClassifier,
                 X_prior: pd.DataFrame,
                 y_prior: pd.Series):
        
        self.__dict__.update(model.__dict__)

        self.X_prior = X_prior
        self.y_prior = y_prior

        self.model = model # Stored for sklearn compatibility

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series):
        return super().fit(pd.concat([self.X_prior, X], ignore_index=True),
                           pd.concat([self.y_prior, y], ignore_index=True))