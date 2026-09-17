import matplotlib.pyplot as plt
import numpy as np


def set_font_sizes(plt):
    fontsizes_normal = 12
    fontsizes_small = 8
    fontsizes_ax = 9
    fontsizes_footnote = 8

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": fontsizes_normal,
            "axes.titlesize": fontsizes_normal,
            "axes.labelsize": fontsizes_ax,
            "legend.fontsize": fontsizes_small,
            "xtick.labelsize": fontsizes_footnote,
            "ytick.labelsize": fontsizes_footnote,
        }
    )


def make_figure(rel_width=1.0, ratio=2):
    total_width = 5.5
    w = total_width * rel_width
    h = w / ratio
    return plt.figure(figsize=(w, h))


def make_subplots(nrows=1, ncols=1, rel_width=1.0, ratio=2, **kwargs):
    total_width = 5.5
    w = total_width * rel_width
    h = w / (ncols * ratio)

    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)
    return fig, axes


def theory_global(tau):
    """Theoretical asymptotic limit for Standard GD"""
    res = np.zeros_like(tau)
    mask1 = tau <= 1.0
    mask2 = (tau > 1.0) & (tau <= 2.0)
    res[mask1] = 1.0 - tau[mask1] ** 2 + (tau[mask1] ** 3) / 3.0
    res[mask2] = (2.0 - tau[mask2]) ** 3 / 3.0
    return res


def theory_row_specific(tau, u_req):
    res = np.zeros_like(tau)
    for k, t_val in enumerate(tau):
        if t_val <= u_req:
            res[k] = 1.0
        elif t_val <= u_req + 1.0:
            res[k] = (1.0 - (t_val - u_req)) ** 2
        else:
            res[k] = 0.0
    return res
