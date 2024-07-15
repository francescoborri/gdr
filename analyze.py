#! /usr/bin/env python3

import argparse
import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import ConciseDateFormatter
from numpy.fft import fft, fftfreq
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.stattools import adfuller

from rrd import rrd_fetch
from utils import timedelta_type


def plot_fft(series: pd.Series, title: str = "FFT", top: int = 3) -> tuple[plt.Figure, list[timedelta]]:
    """
    This functions plots the Fourier transform for the provided time series and returns the frequencies that maximize the amplitude.

    Parameters
    ---
    series : `pandas.Series`
        The time series.
    title : `str, optional`
        The title of the plot.
    top : `int, optional`
        The number of frequencies to display and return.

    Return
    ---
    fig : `matplotlib.figure.Figure`
        The figure containing the plot.
    best_periods : `list[datetime.timedelta]`
        The list of periods corresponding to the best frequencies.
    """

    step = pd.Timedelta(series.index.freq).to_pytimedelta().total_seconds()

    f = pd.Series(abs(fft(series)), index=fftfreq(len(series), d=step))
    f = f[f.index > 0]

    freqs = f.nlargest(top).index
    periods = 1 / freqs
    periods = [timedelta(seconds=period) for period in periods]

    fig, ax = plt.subplots()
    ax.plot(f, color="black")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, np.max(freqs) * 1.1)
    ax.set_title(title)

    for i in range(len(freqs)):
        ax.axvline(freqs[i], color=np.random.rand(3), linestyle="--", label=f"Period {periods[i]}")

    ax.legend(loc="best")

    return fig, periods


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse RRD data with FFT, ACF and PACF plots, ADF test and seasonality decomposition"
    )
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
    parser.add_argument("-d", "--diff", type=int, default=0, metavar="D", help="differencing order")
    parser.add_argument("-D", "--diff_seasonal", type=int, default=0, metavar="D", help="seasonal differencing order")
    parser.add_argument(
        "-m",
        "--periods",
        type=timedelta_type,
        nargs="+",
        metavar="PERIOD",
        help="list of periods for seasonality decomposition (parsed by pandas.Timedelta, see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html for the available formats)",
    )
    parser.add_argument("-w", "--save", type=str, metavar="DIR", help="enables saving plots to a directory")

    args = parser.parse_args()
    start, end, step, data = rrd_fetch(filename=args.filename.name, start=args.start, end=args.end, step=args.step)

    if args.periods:
        for seasonal_period in args.periods:
            if seasonal_period.total_seconds() / step.total_seconds() < 2:
                parser.error(f"A period of {seasonal_period} is too short as step is {step}")
            elif seasonal_period.total_seconds() % step.total_seconds() != 0:
                parser.error(f"A period of {seasonal_period} is not a multiple of the step")

    if args.save:
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        if not os.path.isdir(args.save):
            parser.error(f"{args.save} is not a directory")

    figs = {}

    for source in data:
        series = data[source]

        figs["original"], ax = plt.subplots()
        ax.plot(series, label=f"{source}", color="black")
        ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend(loc="best")

        series.interpolate(method="time", inplace=True)

        figs["fft"], periods = plot_fft(series, title=f"{source} FFT")

        differenced = series
        seasonal_period = int((args.periods[0] if args.periods else periods[0]).total_seconds() // step.total_seconds())

        if args.diff != 0 or args.diff_seasonal != 0:
            differenced = diff(series, k_diff=args.diff, k_seasonal_diff=args.diff_seasonal, seasonal_periods=seasonal_period)
            figs["differenced"], ax = plt.subplots()
            ax.plot(differenced, label=f"{source} d={args.diff},D={args.diff_seasonal}", color="black")
            ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
            ax.legend(loc="best")

        # Null hypothesis: the time series is non-stationary
        _, pvalue = adfuller(differenced)[:2]
        print(f"ADF test p-value={pvalue} => the series is likely {'non-' if pvalue > 0.05 else ''}stationary")

        figs["acf"] = plot_acf(differenced, title=f"{source} ACF", lags=seasonal_period)
        figs["pacf"] = plot_pacf(differenced, title=f"{source} PACF", lags=seasonal_period)

        if args.save:
            for name, fig in figs.items():
                fig.set_size_inches(19.20, 10.80)
                fig.savefig(os.path.join(args.save, f"{source}-{name}.png"), dpi=100)

    if not args.save:
        plt.show()
