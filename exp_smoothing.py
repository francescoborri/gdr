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
    parser.add_argument("filename", type=argparse.FileType("r"), help="input RRD file")
    parser.add_argument(
        "-s",
        "--start",
        type=str,
        default="end-30d",
        metavar="START",
        help="start time from which fetch data (parsed by rrdtool using the AT-STYLE format), default is 30 days before the last observation in the file",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=str,
        default="last",
        metavar="END",
        help="end time until which fetch data (parsed by rrdtool using the AT-STYLE format), default is the last observation in the file",
    )
    parser.add_argument(
        "-i",
        "--step",
        type=timedelta_type,
        default=None,
        metavar="STEP",
        help="preferred interval between 2 data points (note: if specified the data may be downsampled)",
    )
    parser.add_argument(
        "-m",
        "--seasonal_period",
        type=timedelta_type,
        default=timedelta(days=1),
        metavar="SEAS_PERIOD",
        help="seasonal period (parsed by pandas.Timedelta, see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html for the available formats), default is 1 day.",
    )
    parser.add_argument(
        "-f",
        "--forecast_period",
        type=timedelta_type,
        default=timedelta(days=1),
        metavar="FC_PERIOD",
        help="forecast period (parsed the same way as seasonal period), default is 1 day.",
    )
    parser.add_argument(
        "-t",
        "--trend_type",
        type=str,
        default="add",
        choices=["add", "mul", "additive", "multiplicative"],
        help="trend type for the Holt-Winters method",
    )
    parser.add_argument(
        "-l",
        "--seasonal_type",
        type=str,
        default="add",
        choices=["add", "mul", "additive", "multiplicative"],
        help="seasonal type for the Holt-Winters method",
    )
    parser.add_argument(
        "-o",
        "--output_filename",
        type=argparse.FileType("w"),
        metavar="OUT",
        help="optional CSV output filename for the forecasted values",
    )

    args = parser.parse_args()
    start, end, step, data = rrd_fetch(filename=args.filename.name, start=args.start, end=args.end, step=args.step)

    season_offset = int(args.seasonal_period.total_seconds() // step.total_seconds())
    if season_offset < 2:
        parser.error(f"Seasonal period is too short since step is {step} and seasonal period is {args.seasonal_period}")
    elif args.seasonal_period.total_seconds() % step.total_seconds() != 0:
        parser.error("Seasonal period is not a multiple of the step")

    if args.output_filename:
        print("ds,timestamp,value", file=args.output_filename)

    for source in data:
        series = data[source]

        nan_indexes = series[series.isna()].index
        series = series.interpolate(method="time")

        fit = ExponentialSmoothing(
            endog=series, trend=args.trend_type, seasonal=args.seasonal_type, seasonal_periods=season_offset
        ).fit()
        prediction = fit.predict(start=start, end=end + args.forecast_period - step)

        print(fit.summary())

        fig, ax = plt.subplots()
        ax.plot(series, color="black", label="Observed")
        ax.plot(nan_indexes, series[nan_indexes], "o", color="red", label="Interpolated data")
        ax.plot(prediction, color="blue", linestyle="--", label="Prediction")
        ax.axvline(x=end - step, color="black", linestyle=":", label="Last observation")
        ax.set_xlabel("Time")
        ax.set_ylabel(source)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend(loc="best")
        fig.autofmt_xdate()

        if args.output_filename:
            for dt, value in prediction[prediction.index >= end].items():
                print(f"{source},{int(dt.timestamp())},{value}", file=args.output_filename)

    plt.show()
