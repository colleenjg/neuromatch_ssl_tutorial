from matplotlib import pyplot as plt
import numpy as np


def plot_dsprites_images(images, ncols=5, title=None):
    """
    plot_dsprites_images(images)

    Plots dSprites images.

    Required args:
    - images (array-like): list or array of images (allows None values to skip subplots)

    Optional args:
    - ncols (int): maximum number of columns. (default: 5)
    - title (str): plot title. If None, no title is included. (default: None)

    Returns:
    - fig (plt.Figure): figure
    - axes (plt.Axes): axes
    """

    num_images = len(images)
    ncols = np.min([num_images, ncols])
    nrows = int(np.ceil(num_images / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
        figsize=(ncols * 2.2, nrows * 2.2), squeeze=False
        )

    if title is not None:
        fig.suptitle(title, y=1.04)

    for ax_i, ax in enumerate(axes.flatten()):
        if images[ax_i] is not None and ax_i < num_images:
            ax.imshow(images[ax_i], cmap='Greys_r',  interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    return fig, axes


def plot_rsms(rsms, titles=None):
    """
    plot_rsms(rsms)

    Plots representation similarity matrices.

    Required args:
    - rsms (list): list of 2D RSMs arrays.

    Optional args:
    - titles (list): title for each RSM. (default: None)

    Returns:
    - fig (plt.Figure): figure
    - axes (plt.Axes): axes
    """

    if not isinstance(rsms, list):
        rsms = [rsms]
        titles = [titles]

    if len(rsms) != len(titles):
        raise ValueError("If providing titles, must provide as many "
            "as the number of RSMs.")

    min_val = np.min([rsm.min() for rsm in rsms])
    max_val = np.max([rsm.max() for rsm in rsms])
    ncols = len(rsms)
    wid = 6

    fig, axes = plt.subplots(ncols=ncols, figsize=[ncols * wid, wid], squeeze=False)
    fig.suptitle("Representation Similarity Matrices (RSMs)", y=1.05)

    cm_w = 0.05 / ncols
    fig.subplots_adjust(right=1-cm_w*2)
    cbar_ax = fig.add_axes([1, 0.15, cm_w, 0.7])

    for ax, rsm, title in zip(axes.flatten(), rsms, titles):
        im = ax.imshow(rsm, vmin=min_val, vmax=max_val, interpolation="none")
        ax.set_title(title, y=1.02)

    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(label="Similarity", size=18)
    cbar_ax.yaxis.set_label_position("left")

    return fig, axes

