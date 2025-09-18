from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score
from model import ResearchModel


class ArimaResearchModel(ResearchModel):
    def __init__(self, order=(5, 1, 0)):
        super().__init__()
        self.order = order
        self.model = None
        self.fitted_ = False

    def fit(self, Ytrain):
        self.model = ARIMA(Ytrain, order=self.order)
        self.model = self.model.fit()
        self.fitted_ = True
        return self

    def forward(self, X):
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(start=0, end=len(X)-1)
