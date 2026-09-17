import matplotlib.pyplot as plt
import numpy as np


_GOLDEN_RATIO = (1 + 5.0 ** (1 / 2)) / 2.0
_INVERSE_GOLDEN_RATIO = _GOLDEN_RATIO - 1


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


def make_subplots_on_figure(fig, nrows, ncols, **kwargs):
    return fig.subplots(nrows=nrows, ncols=ncols, **kwargs)


def make_subplots(nrows=1, ncols=1, rel_width=1.0, ratio=2, **kwargs):
    fig = make_figure(rel_width=rel_width, ratio=ncols * ratio)
    return fig, make_subplots_on_figure(fig, nrows, ncols, **kwargs)


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


def matplotlib_config(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=4,
    height_to_width_ratio=_INVERSE_GOLDEN_RATIO,
    dpi=250,
):
    return {
        **font_config(),
        **fontsize_config(),
        **layout_config(rel_width, ncols, nrows, height_to_width_ratio, dpi),
        **style_config(),
    }


def fontsize_config():
    fontsizes_normal = 11 - 1
    fontsizes_small = 11 - 3
    fontsizes_tiny = 11 - 4
    return {
        "font.size": fontsizes_normal,
        "axes.titlesize": fontsizes_normal,
        "axes.labelsize": fontsizes_small,
        "legend.fontsize": fontsizes_small,
        "xtick.labelsize": fontsizes_tiny,
        "ytick.labelsize": fontsizes_tiny,
    }


def layout_config(rel_width, ncols, nrows, height_to_width_ratio, dpi):
    full_width_in = 5.5
    width_in = full_width_in * rel_width
    subplot_width_in = width_in / ncols
    subplot_height_in = height_to_width_ratio * subplot_width_in
    height_in = subplot_height_in * nrows
    return {
        "figure.dpi": dpi,
        "figure.figsize": (width_in, height_in),
        "figure.constrained_layout.use": False,
        "figure.autolayout": False,
        # Padding around axes objects. Float representing inches.
        # Default is 3/72 inches (3 points)
        "figure.constrained_layout.h_pad": (1 / 72),
        "figure.constrained_layout.w_pad": (1 / 72),
        # Space between subplot groups. Float representing
        # a fraction of the subplot widths being separated.
        "figure.constrained_layout.hspace": 0.00,
        "figure.constrained_layout.wspace": 0.00,
    }


def style_config():
    return {
        "axes.labelpad": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "ytick.major.pad": 1,
        "xtick.major.pad": 1,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        "axes.titlepad": 3,
    }


def font_config():
    return {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": "\\usepackage{times} ",
    }


def update_style(
    plt,
    rel_width=1.0,
    nrows=1,
    ncols=4,
    height_to_width_ratio=_INVERSE_GOLDEN_RATIO,
    dpi=250,
):
    plt.rcParams.update(
        matplotlib_config(
            rel_width=rel_width,
            nrows=nrows,
            ncols=ncols,
            height_to_width_ratio=height_to_width_ratio,
            dpi=dpi,
        )
    )


def nice_logspace(start, stop, density, base=10):
    """Returns a log-spaced grid between base**start and base**end

    Increasing the density will repeat previously hit values

    Plays nicely with ``merge_grids`` to merge a sparse and a dense grid
    ``merge_grids(nice_logspace(-4, 3, density=1), nice_logspace(-2, 0, density=2)``

    Start, end and density are assumed to be integers
    Density = 1 will return (end - start) points
    Increasing density by 1 doubles the number of points
    """
    if density < 1 or not np.allclose(int(density), density):
        raise ValueError(f"Density needs to be an integer >= 1, got {density}.")
    if not np.allclose(int(start), start) or not np.allclose(int(stop), stop):
        raise ValueError(f"Start and end need to be integers, got {start, stop}.")
    if not (stop > start):
        raise ValueError(f"Start needs to be smaller than stop, got {start, stop}.")
    assert stop > start
    return np.logspace(
        start, stop, base=base, num=(stop - start) * (2 ** (density - 1)) + 1
    )
