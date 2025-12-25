import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from river.tree import ExtremelyFastDecisionTreeClassifier

from river.stream import iter_pandas

class ExtremelyFastDecisionTreeWrapper(DecisionTreeClassifier):
    """
    A wrapper for the ExtremelyFastDecisionTreeClassifier from the river library to make it compatible with sklearn's DecisionTreeClassifier interface.
    """
    def __init__(self,
                 X_prior: pd.DataFrame = None,
                 y_prior: pd.Series = None,
                 **kwargs):
        super().__init__()
        
        assert (X_prior is None) == (y_prior is None), "X_pretrain and y_pretrain must be both provided or None"
        
        if "nominal_attributes" not in kwargs:
            assert X_prior is not None, "X_prior must be provided to infer nominal attributes in case not specified"
            kwargs["nominal_attributes"] = list(filter(lambda column_name: X_prior[column_name].dtype in [object, bool], X_prior.columns))

        self.model = ExtremelyFastDecisionTreeClassifier(**kwargs)

        if X_prior is not None:
            self.fit(X_prior, y_prior)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the ExtremelyFastDecisionTreeClassifier model.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target labels.

        Returns:
            ExtremelyFastDecisionTreeWrapper: The fitted model.
        """
        for x_i, y_i in iter_pandas(X, y):
            self.model.learn_one(x_i, y_i)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the fitted ExtremelyFastDecisionTreeClassifier model.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            pd.Series: The predicted labels.
        """
        predictions = []
        for x_i, _ in iter_pandas(X):
            predictions.append(self.model.predict_one(x_i))
        return np.array(predictions)