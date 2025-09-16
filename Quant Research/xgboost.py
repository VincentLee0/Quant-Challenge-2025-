from xgboost import XGBRegressor
from model import ResearchModel
from sklearn.metrics import mean_squared_error

class XGBoostResearchModel(ResearchModel):
    def __init__(
        self,
        *,
        objective: str = "reg:squarederror",
        n_estimators: int = 2000,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 42,
        early_stopping_rounds: int = 100,
        tree_method = "hist"
    ):
        super().__init__()
        self.model = XGBRegressor(
            objective=objective,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            tree_method=tree_method,  # fast default; can be 'gpu_hist' if GPU available
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.fitted_ = False
        self.feature_names_ = None

    def fit(self, Xtrain, Ytrain, Xvalid=None, Yvalid=None):
        """
        Train the XGBoost regressor.
        - Xtrain: 2D array or DataFrame of engineered features
        - Ytrain: 1D array/Series of target values
        - Xvalid/Yvalid: optional validation set for early stopping

        If no validation passed, the model trains to n_estimators without early stopping.
        """
        eval_set = None
        if Xvalid is not None and Yvalid is not None:
            eval_set = [(Xvalid, Yvalid)]

        self.model.fit(
            Xtrain,
            Ytrain,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
        )

        self.fitted_ = True
        if hasattr(Xtrain, "columns"):
            self.feature_names_ = list(Xtrain.columns)
        else:
            self.feature_names_ = None

        return self

    def forward(self, X):
        """
        Inference. Returns predictions for provided feature matrix.
        This expects the same feature engineering as used in fit().
        """
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)
