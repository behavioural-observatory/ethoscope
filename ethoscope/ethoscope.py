# Pranav Minasandra
# pminasandra.github.io
# 3 Jun 2026

import uuid

import accutils as au
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

from . import config

def find_thresholds(log_vedba_data, prominence=0.01, make_plots=False, fname=None):
    """
    Returns thresholds based on VeDBA histograms using a Gaussian KDE and a peak-finding
    approach.
    Args:
        log_vedba_data (np.array): all log(VeDBA) data meaned over desired intervals.
        make_plots (bool): whether to store a plot showing performance
        fname (str | None): only if make_plots is True, the name of the file.
    Returns:
        np.array: each threshold value
    """

    data = log_vedba_data.copy()
    data = data[~np.isnan(data)]
    kde = gaussian_kde(data)
    left_border = np.quantile(data, q=config.LEFT_QUANTILE)
    right_border = np.quantile(data, q=config.RIGHT_QUANTILE)
    x = np.linspace(left_border, right_border, 5_000)
    y = kde(x)

    troughs, properties = find_peaks(-y, prominence=prominence)

    if make_plots:
        if fname is None:
            fname = f"{species}_{uuid.uuid()}"
        fig, ax =plt.subplots(constrained_layout=True)
        ax.hist(data, 200, density=True)
        ax.plot(x, y, color="black")
        ymin, ymax = ax.get_ylim()
        ax.vlines(x[troughs], color="red", linestyle="dotted",
                    linewidth=0.7, ymin=ymin, ymax=ymax)
        fig.savefig(fname)
    return x[troughs]

def classify_states(log_vedba_data, make_plots=False, fname=None):
    """
    Based on VeDBA data, ascribes an activity level to each variable
    Args:
        log_vedba_data (np.array): all log(VeDBA) data meaned over desired intervals.
    Returns:
        np.array: same shape as log_vedba_data, with activity levels 0, 1, ...
        int: number of detected states
    """
    thresholds = find_thresholds(log_vedba_data, make_plots=make_plots, fname=fname)
    out = np.empty(shape=log_vedba_data.shape)
    out[:] = np.nan
    nonnan = ~np.isnan(log_vedba_data)
    out[nonnan] = np.digitize(log_vedba_data[nonnan], thresholds)

    return out, len(np.unique(out[nonnan])), thresholds

def parse(accdf, make_plots=False, window='2s',
            retain_vedba_col=False, au_kw={}, fname=None):
    """
    Wrapper, takes in an acc dataframe and returns a behavioural sequence.
    Args:
        accdf (pd.DataFrame)
        au_kw (dict): arguments passed to the accutils.secondwise_mean(...) function.
        retain_vedba_col (bool): whether vedba_col should be retained
    Returns:
        pd.DataFrame: behavioural sequence in PNAS paper format
        dict: containing keys "n_states" (int) and "thresholds" (list).
    """

    df = accdf.copy()
    df = au.filters.smudge_acc_data(df)
    df.loc[:, "vedba"] = au.feature_ext.dbas.vedba_vals(df, window=window)
    df = au.feature_ext.secondwise_mean(df, window=window)

    lvdb = np.log(df["vedba_mean"])
    beh_seq, n_states, thresholds = classify_states(lvdb, make_plots=make_plots, fname=fname)
    df["state"] = beh_seq
    df["datetime"] = df[au.Timestamp]
    if not retain_vedba_col:
        df = df[["datetime", "state"]]
    else:
        df = df[["datetime", "state", "vedba_mean"]]

    return df, {"n_states": n_states, "thresholds": list(thresholds)}
