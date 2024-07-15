from datetime import timedelta

import pandas as pd


def timedelta_type(period: str) -> timedelta:
    """
    This function parses a string representing a period and returns the corresponing `datetime.timedelta` object.

    Parameters
    ---
    period : `str`
        A string representing a period according to the `pandas.Timedelta` formats.

    Return
    ---
    timedelta : `datetime.timedelta`
        A `datetime.timedelta` object corresponding to the provided period.

    """

    return pd.Timedelta(period).to_pytimedelta()
