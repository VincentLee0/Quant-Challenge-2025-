import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib
from model import ResearchModel

class RandomForestResearchModel(ResearchModel):
    def __init__(self, n_estimators = 100, max_depth = None, min_samples_split = 2, random_state =  42, n_jobs = 1):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.models = {}
        self.feature_columns = None
        self.target_columns = None

    def fit(self, Xtrain, Ytrain):
        self.feature_columns = Xtrain.columns.tolist()
        self.target_columns = Ytrain.columns.tolist()

        for target in self.target_columns:
            print(f"Training random forest for target: {target}")

            # Remove rows where target is missing

            valid_idx = Ytrain[target].notna()
            X_train_valid = Xtrain[valid_idx]
            y_train_valid = Ytrain.loc[valid_idx, target]

            model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('rf', RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split = self.min_samples_split,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs

                ))
            ])

            #TODO add cross-validation here to monitor performance

            #Train the model
            model.fit(X_train_valid, y_train_valid)
            self.models[target] = model
    def forward(self, x):
        if not self.models:
            raise ValueError("Model must be fitted before making a prediction")
        predictions = pd.DataFrame(indexx = x.index)
        for target, model in self.model.items():
            predictions[target] = model.predict(x)
        return predictions