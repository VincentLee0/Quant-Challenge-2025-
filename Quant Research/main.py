from data_test import get_processed_stock_data
import pandas as pd
from RandomForest import RandomForestResearchModel
from xgboost import XGB

def process_data():
    column_to_predict = "Close"

    stock_data = get_processed_stock_data()

    XTrain = stock_data.drop(columns=[column_to_predict])
    YTrain = stock_data[column_to_predict]

    return XTrain, YTrain

XTrain, YTrain = process_data()

model = RandomForestResearchModel()

model.fit(XTrain, YTrain)