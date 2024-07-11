#! /usr/bin/env python3

import argparse
from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from rrd import rrd_fetch
from utils import timedelta_type


def order_type(order: str):
    t = tuple(map(int, order.split(",")))

    if len(t) != 3:
        raise argparse.ArgumentTypeError("Order must be a 3-tuple")

    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecasting")
    parser.add_argument("filename", type=argparse.FileType("r"), help="Input RRD file")
    parser.add_argument("--order", type=int, help="ARIMA order", nargs=3, metavar=("p", "d", "q"))
    parser.add_argument("--start", type=str, help="Start time", default="end-30d")
    parser.add_argument("--end", type=str, help="End time", default="last")
    parser.add_argument("--forecast_period", type=timedelta_type, help="Forecast period", default=timedelta(days=1))
    parser.add_argument("--output_filename", type=argparse.FileType("w"), help="CSV output filename")

    args = parser.parse_args()
    (start, end, step), data = rrd_fetch(args.filename.name, args.start, args.end)

    if args.output_filename:
        print("ds,timestamp,value", file=args.output_filename)

    for source in data:
        series = data[source].interpolate(method="time")

        fit = ARIMA(endog=series, order=args.order, trend="n").fit()
        print(fit.summary())

        forecast = pd.Series(
            fit.predict(end, end + args.forecast_period),
            index=pd.date_range(end, end + args.forecast_period, freq=step),
        )

        fig, ax = plt.subplots()
        ax.plot(series, color="black", label="Observed")
        ax.plot(forecast, color="blue", linestyle="--", label="Forecast")

        ax.set_xlabel("Time")
        ax.set_ylabel(source)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend(loc="best")
        fig.autofmt_xdate()

        if args.output_filename:
            for timestamp, value in forecast.items():
                print(f"{source},{timestamp},{value}", file=args.output_filename)

    plt.show()
