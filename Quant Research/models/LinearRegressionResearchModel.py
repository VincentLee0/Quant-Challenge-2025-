from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from model import ResearchModel
from graph import plot


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
            Ypred = self.forward(Xval)
            self.validation_score_ = r2_score(Yval, Ypred)
            plot(Yval.tolist(), Ypred.tolist(), "LR Validation Results")
            print(f"Validation R² score: {self.validation_score_:.4f}")
        else:
            self.validation_score_ = None

        return self

    def forward(self, X):
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)
