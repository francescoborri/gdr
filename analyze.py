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
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.stattools import adfuller

from rrd import rrd_fetch
from utils import timedelta_type


def plot_fft(series, title="FFT", n=5):
    step = pd.Timedelta(series.index.freq).to_pytimedelta().total_seconds()

    f = pd.Series(abs(fft(series)), index=fftfreq(len(series), d=step))
    f = f[f.index > 0]

    best_freqs = f.nlargest(n).index
    best_periods = 1 / best_freqs
    best_periods = [timedelta(seconds=period) for period in best_periods]

    fig, ax = plt.subplots()
    ax.plot(f, color="black")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, np.max(best_freqs) * 1.1)
    ax.set_title(title)

    for i in range(len(best_freqs)):
        ax.axvline(best_freqs[i], color=np.random.rand(3), linestyle="--", label=f"Period {best_periods[i]}")

    ax.legend(loc="best")

    return fig, best_periods


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecasting")
    parser.add_argument("filename", type=argparse.FileType("r"), help="Input RRD file")
    parser.add_argument("--start", type=str, help="Start time", default="first")
    parser.add_argument("--end", type=str, help="End time", default="last")
    parser.add_argument("--diff", type=int, help="Differencing order", default=0, metavar="d")
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Enable seasonality decomposition (ACF and PACF will be calculated on residuals)",
    )
    parser.add_argument(
        "--periods",
        type=timedelta_type,
        help="List of periods for seasonality decomposition",
        nargs="+",
        default=[],
    )
    parser.add_argument("--save", type=str, help="Save plots in the specified directory")

    args = parser.parse_args()
    (start, end, step), data = rrd_fetch(args.filename.name, args.start, args.end)

    figs = {}

    for source in data:
        series = data[source]

        figs["original"], ax = plt.subplots()
        ax.plot(series, label=f"{source}", color="black")
        ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend(loc="best")

        series.interpolate(method="time", inplace=True)

        figs["fft"], periods = plot_fft(series, title=f"{source} FFT")
        print("Suggested periods:\n\t" + "\n\t".join(str(period) for period in periods))

        if args.diff != 0:
            series = diff(series, k_diff=args.diff)
            figs["differenced"], ax = plt.subplots()
            ax.plot(series, label=f"{source} differenced {args.diff} times", color="black")
            ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
            ax.legend(loc="best")

        _, pvalue = adfuller(series)[:2]
        print(f"ADF test p-value={pvalue}")

        if pvalue < 0.05:
            print(f"Suggested d={args.diff}")
        else:
            print(f"Suggested d>={args.diff + 1}")

        if args.decompose:
            periods = [
                int(period.total_seconds() // step.total_seconds())
                for period in (args.periods if len(args.periods) > 0 else periods)
            ]
            decomposition = MSTL(series, periods=periods).fit()
            figs["decomposition"] = decomposition.plot()

            series = decomposition.resid

        figs["acf"] = plot_acf(series, title=f"{source} ACF")
        figs["pacf"] = plot_pacf(series, title=f"{source} PACF")

        if args.save:
            if not os.path.exists(args.save):
                os.makedirs(args.save)

            if not os.path.isdir(args.save):
                raise ValueError(f"{args.save} is not a directory")

            for name, fig in figs.items():
                fig.savefig(f"{args.save}/{source}-{name}.png")

    if not args.save:
        plt.show()
