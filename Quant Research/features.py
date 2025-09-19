import pandas as pd
import pandas_ta as ta


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

    def add_momentum_indicators(self):
        self.df['RSI_14'] = ta.rsi(self.df['Close'], length=14)

        macd = ta.macd(self.df['Close'], fast=12, slow=26, signal=9)
        self.df[['MACD_12_26_9', 'MACD_signal', 'MACD_hist']] = macd
        return self

    def add_volatility_indicators(self):
        self.df['ATR_14'] = ta.atr(
            self.df['High'], self.df['Low'], self.df['Close'], length=14)

        return self

    def add_trend_indicators(self):
        self.df['SMA_20'] = ta.sma(self.df['Close'], length=20)
        self.df['SMA_100'] = ta.sma(self.df['Close'], length=100)
        self.df['SMA_ratio'] = self.df['SMA_20'] / self.df['SMA_100']

        return self

    def generate_features(self):
        self.add_momentum_indicators()
        self.add_volatility_indicators()
        self.add_trend_indicators()

        return self

    def get_dataframe(self):
        return self.df
