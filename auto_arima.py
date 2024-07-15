#! /usr/bin/env python3

import argparse
import os
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.arima import ARIMASummary
from statsforecast.models import AutoARIMA

from rrd import rrd_fetch
from utils import timedelta_type

os.environ["NIXTLA_ID_AS_COL"] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARIMA forecast with automatic order selection")
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
        "-o",
        "--output_filename",
        type=argparse.FileType("w"),
        metavar="OUT",
        help="optional CSV output filename for the forecasted values",
    )

    args = parser.parse_args()
    start, end, step, data = rrd_fetch(filename=args.filename.name, start=args.start, end=args.end, step=args.step)

    season_offset = int(args.seasonal_period.total_seconds() // step.total_seconds()) if args.seasonal_period else 0
    forecast_offset = int(args.forecast_period.total_seconds() // step.total_seconds())

    if args.seasonal_period and season_offset < 2:
        parser.error(f"Seasonal period is too short since step is {step} and seasonal period is {args.seasonal_period}")
    elif args.seasonal_period.total_seconds() % step.total_seconds() != 0:
        parser.error("Seasonal period is not a multiple of the step")

    if args.output_filename:
        print("ds,timestamp,value", file=args.output_filename)

    for source in data:
        series = data[source]

        nan_indexes = series.index[series.isna()]
        series = series.interpolate(method="time")

        datestamps = [dt.astype(datetime) for dt in series.index.values]
        y = series.values
        unique_id = 1
        df = pd.DataFrame({"ds": datestamps, "y": y, "unique_id": unique_id})

        model = AutoARIMA(season_length=season_offset) if season_offset > 0 else AutoARIMA()
        sf = StatsForecast(models=[model], freq=int(step.total_seconds()), n_jobs=-1).fit(df=df)

        fitted_model = sf.fitted_[0, 0].model_
        season_offset = fitted_model["arma"][4]

        print("Summary:")
        ARIMASummary(fitted_model).summary()
        if season_offset > 1 and not args.seasonal_period:
            print(
                f"A seasonal period of {season_offset} corresponds to {timedelta(seconds=season_offset * step.total_seconds())}"
            )

        levels = [25, 50, 75]

        prediction_result = sf.forecast(h=forecast_offset, level=levels, fitted=True)
        prediction_index = pd.date_range(end, end + args.forecast_period, freq=step, inclusive="left")

        fitted_values = pd.Series(sf.forecast_fitted_values()["AutoARIMA"].values, index=series.index)
        prediction = pd.Series(
            sf.forecast(h=forecast_offset, level=levels, fitted=True)["AutoARIMA"].values,
            index=prediction_index,
        )

        prediction = pd.concat([fitted_values, prediction])

        fig, ax = plt.subplots()
        ax.plot(series, color="black", label="Observed")
        ax.plot(nan_indexes, series[nan_indexes], "o", color="red", label="Interpolated data")
        ax.plot(prediction, color="blue", linestyle="--", label="Prediction")
        ax.axvline(x=end - step, color="black", linestyle=":", label="Last observation")

        for level in levels:
            ax.fill_between(
                prediction_index,
                prediction_result[f"AutoARIMA-lo-{level}"],
                prediction_result[f"AutoARIMA-hi-{level}"],
                color="orange",
                alpha=1 - level / 100,
                label=f"{level}% confidence interval",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel(source)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend(loc="best")
        fig.autofmt_xdate()

        if args.output_filename:
            for dt, value in prediction.items():
                print(f"{source},{int(dt.timestamp())},{value}", file=args.output_filename)

    plt.show()
