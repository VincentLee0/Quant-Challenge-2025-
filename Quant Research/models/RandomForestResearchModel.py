from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from model import ResearchModel
from graph import plot
import numpy as np
import pandas as pd

class RandomForestResearchModel(ResearchModel):
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: str = "sqrt",
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: int = None,
    ):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
        self.fitted_ = False
        self.feature_names_ = None

    def fit(self, Xtrain, Ytrain, Xval=None, Yval=None):
        self.model.fit(Xtrain, Ytrain)
        
        self.fitted_ = True
        if hasattr(Xtrain, "columns"):
            self.feature_names_ = list(Xtrain.columns)
        else:
            self.feature_names_ = None
            
        # Calculate validation score if validation data is provided
        if Xval is not None and Yval is not None:
            y_pred = self.forward(Xval)
            from sklearn.metrics import r2_score
            self.validation_score_ = r2_score(Yval, y_pred)
            plot(Yval.tolist(), y_pred.tolist(), "RF Validation Results", self.validation_score_)
            print(f"Validation R² score: {self.validation_score_:.4f}")
        else:
            self.validation_score_ = None
            
        return self

    def forward(self, x):
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(x)