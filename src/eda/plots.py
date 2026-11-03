import matplotlib.pyplot as plt


def set_labels(ax, title, xlabel=None, ylabel=None):
    """
    Sets the title (size 12, bold) and the axis labels of one plot.
    Labels left as None are not changed.

    Parameters:
        ax     -> the plot (e.g. axes[0] or axes[1, 0])
        title  -> title of the plot
        xlabel -> label of the x axis (optional)
        ylabel -> label of the y axis (optional)
    """
    ax.set_title(title, fontsize=12, fontweight='bold')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

#-----------------------------------------------------------------------------------------#

def finish_figure(title, y=1.02):
    """
    Adds the main title of the figure (size 15, bold), adjusts the layout and shows the figure.

    Parameters:
        title -> main title of the figure
        y     -> vertical position of the title (default 1.02; higher = further up)
    """
    plt.suptitle(title, fontsize=15, fontweight='bold', y=y)
    plt.tight_layout()
    plt.show()
