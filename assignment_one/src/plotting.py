import matplotlib

SPINE_COLOR = 'gray'


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    from math import sqrt
    import matplotlib

    assert columns in {1, 2}
    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {
              'text.latex.preamble': [r'\usepackage{microtype}', r'\usepackage{libertine}', r'\usepackage{libertinust1math}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'axes.titlepad' : 20,
              'font.size': 12, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family' : 'serif',
              'lines.markersize' : 2,
              'lines.linewidth' : 0.7,
              'axes.prop_cycle' : "cycler('linestyle', ['solid', 'dashed', 'dashdot', 'dotted'])"
    }

    matplotlib.rcParams.update(params)


def format_axes(ax, SPINE_COLOR = 'gray', NTICKS=4):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.minorticks_off()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    if ax.get_yaxis().get_scale() == 'log':
        y_ticks = matplotlib.ticker.LogLocator(subs='auto', numticks=NTICKS)
        y_ticks_min = matplotlib.ticker.LogLocator()
        ax.yaxis.set_minor_locator(y_ticks_min)
    else:
        y_ticks = matplotlib.ticker.MaxNLocator(NTICKS)
        y_ticks_min = matplotlib.ticker.AutoMinorLocator(2)
        y_format = matplotlib.ticker.ScalarFormatter()
        y_format.set_powerlimits((-2, 2))
        y_format.set_scientific(True)
        ax.yaxis.set_major_formatter(y_format)
    ax.yaxis.set_major_locator(y_ticks)
    ax.yaxis.set_minor_locator(y_ticks_min)
    return ax