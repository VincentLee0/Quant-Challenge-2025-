import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


class LinearRegressionResearchModel(ResearchModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        self.fitted_ = False
        self.feature_names_ = None

    def fit(self, Xtrain, Ytrain, Xval=None, Yval=None):
        self.model.fit(Xtrain, Ytrain)

        self.fitted_ = True
        if hasattr(Xtrain, "columns"):
            self.feature_names_ = list(Xtrain.columns)
        else:
            self.feature_names_ = None

        if Xval is not None and Yval is not None:
            val_score = self.model.score(Xval, Yval)
            print(f"Validation R^2 Score: {val_score:.4f}")
