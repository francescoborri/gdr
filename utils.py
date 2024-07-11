import pandas as pd


def timedelta_type(period: str):
    return pd.Timedelta(period).to_pytimedelta()
