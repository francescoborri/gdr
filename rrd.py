from datetime import datetime, timedelta

import pandas as pd
import rrdtool


def rrd_fetch(
    filename: str, start: str, end: str, step: timedelta = None
) -> tuple[datetime, datetime, timedelta, dict[str, pd.Series]]:
    """
    This function fetches data from an RRD file in a specified time range.

    Parameters
    ---
    filename : `str`
        The name of the RRD file to read.
    start : `str`
        The start time from which fetch data. The datetime formats available are the same of `rrdtool` (AT-STYLE), with the addition of the keyword "last", which means the timestamp of the last observation in the file.
    end : `str`
        The end time from which fetch data. The datetime formats available are the same of the parameter `start`.
    step : `datetime.timedelta`, optional
        The preferred time interval between 2 data points. It must be greater or equal to the resolution of the RRD file.

    Return
    ---
    start : `datetime.datetime`
        The adjusted start time returned from `rrdtool`
    end : `datetime.datetime`
        The adjusted end time returned from `rrdtool`
    step : `datetime.timedelta`
        The time interval between 2 data points
    data : `dict[str, pandas.Series]`
        A dictionary containing the data, indexed by the names of the data sources, with values of type `pandas.Series`.
    """

    last_ts = rrdtool.last(filename)

    if "last" in end:
        end = end.replace("last", str(last_ts))

    (start, end, rrd_step), ds, raw = rrdtool.fetch(filename, "AVERAGE", "--start", start, "--end", end)

    start = datetime.fromtimestamp(start)
    rrd_step = timedelta(seconds=rrd_step)
    end = start + len(raw) * rrd_step
    zipped = dict(zip(ds, zip(*raw)))

    data = {}

    for source in zipped:
        index = pd.date_range(start, end, freq=rrd_step, inclusive="left")
        data[source] = pd.Series(zipped[source], index=index)

        if step:
            if step < rrd_step:
                raise ValueError(f"Cannot downsample from {rrd_step} to {step}")
            else:
                data[source] = data[source].resample(step).mean()
                start = data[source].index[0]
                end = data[source].index[-1] + step
        else:
            step = rrd_step

    return start, end, step, data
