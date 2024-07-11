#! /usr/bin/env python3

import argparse
from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from rrd import rrd_fetch
from utils import timedelta_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exponential smoothing forecast")
    parser.add_argument("filename", type=argparse.FileType("r"), help="Input RRD file")
    parser.add_argument("--start", type=str, help="Start time", default="end-30d")
    parser.add_argument("--end", type=str, help="End time", default="last")
    parser.add_argument("--seasonal_period", type=timedelta_type, help="Seasonal period", default=timedelta(days=1))
    parser.add_argument("--forecast_period", type=timedelta_type, help="Forecast period", default=timedelta(days=1))
    parser.add_argument("--output_filename", type=argparse.FileType("w"), help="Output filename", required=False)

    args = parser.parse_args()
    (start, end, step), data = rrd_fetch(args.filename.name, args.start, args.end)

    if args.output_filename:
        print("ds,timestamp,value", file=args.output_filename)

    for source in data:
        series = data[source]

        nan_idxs = series[series.isna()].index
        filled_series = series.interpolate(method="time")

        fig, ax = plt.subplots()
        ax.plot(series, color="black", label="Original observed")
        ax.plot(nan_idxs, filled_series[nan_idxs], "o", color="red", label="Interpolated data")
        ax.set_xlabel("Time")
        ax.set_ylabel(source)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend(loc="best")
        fig.autofmt_xdate()

        hwm_fit = ExponentialSmoothing(
            endog=filled_series,
            trend="add",
            seasonal="add",
            seasonal_periods=int(args.seasonal_period.total_seconds() // step.total_seconds()),
        ).fit()
        hwm_forecast = hwm_fit.predict(end, end + args.forecast_period)
        print(hwm_fit.summary())

        fig, ax = plt.subplots()
        ax.plot(filled_series, color="black", label="Observed")
        ax.plot(
            hwm_fit.fittedvalues,
            color="blue",
            linestyle="--",
            label="Holt-Winters method fitted values",
            linewidth=0.5,
        )
        ax.plot(
            hwm_forecast,
            color="blue",
            linestyle="--",
            label=f"Holt-Winters method $\\alpha={hwm_fit.model.params['smoothing_level']:.2f}$, $\\beta={hwm_fit.model.params['smoothing_trend']:.2f}$, $\\gamma={hwm_fit.model.params['smoothing_seasonal']:.2f}$",
        )

        ax.set_xlabel("Time")
        ax.set_ylabel(source)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend(loc="best")
        fig.autofmt_xdate()

        if args.output_filename:
            for timestamp, value in hwm_forecast.items():
                print(f"{source},{timestamp},{value}", file=args.output_filename)

    plt.show()
