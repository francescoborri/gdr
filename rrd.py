import rrdtool
from datetime import datetime, timedelta
import pandas as pd


def rrd_fetch(filename: str, start: str, end: str):
    last_ts = rrdtool.last(filename)

    if "last" in end:
        end = end.replace("last", str(last_ts))

    (start, end, step), ds, raw = rrdtool.fetch(filename, "AVERAGE", "--start", start, "--end", end)

    start = datetime.fromtimestamp(start)
    step = timedelta(seconds=step)
    end = start + len(raw) * step
    zipped = dict(zip(ds, zip(*raw)))
    
    data = {}

    for source in zipped:
        index = pd.date_range(start, end, freq=step, inclusive="left")
        data[source] = pd.Series(zipped[source], index=index)

    return (start, end, step), data