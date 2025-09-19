from xgboost import XGBRegressor
from model import ResearchModel
from sklearn.metrics import r2_score
from graph import plot

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
        self.fitted_ = False
        self.feature_names_ = None

    def fit(self, Xtrain, Ytrain, Xval=None, Yval=None):
        eval_set = None
        if Xval is not None and Yval is not None:
            eval_set = [(Xval, Yval)]

        self.model.fit(
            Xtrain,
            Ytrain,
            eval_set=eval_set,
            verbose=False,
        )

        self.fitted_ = True
        if hasattr(Xtrain, "columns"):
            self.feature_names_ = list(Xtrain.columns)
        else:
            self.feature_names_ = None
        
        Ypred = self.forward(Xval)
        self.validation_score_ = r2_score(Yval, Ypred)
        plot(Yval.tolist(), Ypred.tolist(), "XGB Validation Results")
        print(f"Validation R² score: {self.validation_score_:.4f}")

        return self

    def forward(self, X):
        """
        Inference. Returns predictions for provided feature matrix.
        This expects the same feature engineering as used in fit().
        """
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)
