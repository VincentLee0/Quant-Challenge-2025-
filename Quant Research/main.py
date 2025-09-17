from data_test import get_processed_stock_data
import pandas as pd
import numpy as np
import pandas as pd
from models.RandomForestResearchModel import RandomForestResearchModel
from models.XGBoostResearchModel import XGBoostResearchModel
from models.LinearRegressionResearchModel import LinearRegressionResearchModel

from sklearn.metrics import r2_score


def process_data(
    column_to_predict: str = "Close",
    *,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    date_col: str | None = None,
    purge: int = 0,
):
    # Load and copy to avoid side effects
    df = get_processed_stock_data().copy()

    # Sort chronologically
    if date_col is not None:
        if date_col not in df.columns:
            raise KeyError(
                f"`{date_col}` not found in data columns: {list(df.columns)}")
        df = df.sort_values(date_col).reset_index(drop=True)
    else:
        # Assume the index already represents time; still enforce sort just in case
        df = df.sort_index()

    N = len(df)
    if N < 10:
        raise ValueError(
            f"Not enough rows ({N}) to split into train/val/test.")

    # Normalize proportions if they don't sum to 1.0
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        train_size, val_size, test_size = (
            train_size / total, val_size / total, test_size / total)

    # Compute sizes
    n_train = int(N * train_size)
    n_val = int(N * val_size)
    n_test = N - n_train - n_val

    # Ensure non-empty splits and adjust if needed
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Invalid split sizes for N={N}: train={n_train}, val={n_val}, test={n_test}. "
            "Try adjusting the ratios."
        )

    # Apply purge gap between train and validation to avoid leakage via rolling windows
    # [0 : n_train) -> TRAIN
    # [n_train + purge : n_train + purge + n_val) -> VALIDATION
    # [n_train + purge + n_val : end) -> TEST
    cut_train_end = n_train
    cut_val_start = n_train + purge
    cut_val_end = cut_val_start + n_val

    # If purge eats too many rows, shrink it
    if cut_val_end > N:
        overshoot = cut_val_end - N
        # Try to reduce purge first
        reduce_purge = min(purge, overshoot)
        purge -= reduce_purge
        cut_val_start = n_train + purge
        cut_val_end = cut_val_start + n_val
        if cut_val_end > N:
            # As a last resort, shrink validation size
            n_val = max(1, N - (n_train + purge + 1))
            cut_val_end = cut_val_start + n_val

    # Build X/y
    X = df.drop(columns=[column_to_predict])
    y = df[column_to_predict]

    Xtrain = X.iloc[:cut_train_end]
    Ytrain = np.array(y.iloc[:cut_train_end].values)

    Xval = X.iloc[cut_val_start:cut_val_end]
    Yval = np.array(y.iloc[cut_val_start:cut_val_end].values)

    Xtest = X.iloc[cut_val_end:]
    Ytest = np.array(y.iloc[cut_val_end:].values)

    return Xtrain, Ytrain.reshape(-1), Xval, Yval.reshape(-1), Xtest, Ytest.reshape(-1)


def test_model(model, Xtest, Ytest):
    Ypred = model.forward(Xtest)
    validation_score_ = r2_score(Ytest, Ypred)
    return validation_score_


column_to_predict = "Close"
Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = process_data(
    column_to_predict=column_to_predict)

model1 = RandomForestResearchModel()
model2 = XGBoostResearchModel()
model3 = LinearRegressionResearchModel()

model1.fit(
    Xtrain=Xtrain,
    Ytrain=Ytrain,
    Xval=Xval,
    Yval=Yval
)

model2.fit(
    Xtrain=Xtrain,
    Ytrain=Ytrain,
    Xval=Xval,
    Yval=Yval
)

model3.fit(
    Xtrain=Xtrain,
    Ytrain=Ytrain,
    Xval=Xval,
    Yval=Yval
)

validation_score1 = test_model(model1, Xtest, Ytest)
validation_score2 = test_model(model2, Xtest, Ytest)
validation_score3 = test_model(model3, Xtest, Ytest)
print(f"RF Test R² score: {validation_score1:.4f}")
print(f"XGB Test R² score: {validation_score2:.4f}")
print(f"LR Test R² score: {validation_score3:.4f}")
