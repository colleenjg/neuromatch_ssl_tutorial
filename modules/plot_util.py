from matplotlib import pyplot as plt
import numpy as np


def plot_dsprites_images(images, ncols=5, title=None):
    """
    plot_dsprites_images(images)

    Plots dSprites images.

    Required args:
    - images (array-like): list or array of images (allows None values to 
        skip subplots). If each image has 3 dimensions, the first is assumed 
        to be the channels, and is 
        averaged across.

    Optional args:
    - ncols (int): maximum number of columns. (default: 5)
    - title (str): plot title. If None, no title is included. (default: None)

    Returns:
    - fig (plt.Figure): figure
    - axes (plt.Axes): axes
    """

    # average channel dimension
    for i, image in enumerate(images):
        if image is not None and len(image.shape) == 3:
            images[i] = np.mean(image, axis=0)

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
            ax.imshow(images[ax_i], cmap='Greys_r', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    return fig, axes


def plot_dsprite_image_doubles(images, image_doubles, doubles_str, ncols=5, 
                               title=None):
    """
    plot_dsprite_image_doubles(images, image_doubles, doubles_str)

    Plots dSprite images is sets of 2 rows.

    Required args:
    - images (list): list of images
    - image_doubles (list): list of image doubles (same length as images)
    - doubles_str (str): string that specified what the doubles are.

    Optional args:
    - ncols (int): number of columns. (default: 5)
    - title (str): plot title. If None, no title is included. (default: None)

    Returns:
    - fig (plt.Figure): figure
    - axes (plt.Axes): axes
    """

    if len(images) != len(image_doubles):
        raise ValueError(
            "images and image_doubles must have the same length, but have "
            f"length {len(images)} and {len(image_doubles)}, respectively."
            )

    if not isinstance(images, list) or not isinstance(image_doubles, list):
        raise ValueError("Must pass images and image_doubles as lists.")

    plot_images = []
    ncols = np.min([len(images), ncols])
    n_sets = int(np.ceil(len(images) / ncols))
    for i in range(n_sets):
        extend_images = images[i * ncols : (i + 1) * ncols]
        extend_image_doubles = image_doubles[i * ncols : (i + 1) * ncols]
        padding = [None] * (ncols - len(extend_images))

        plot_images.extend(
            extend_images + padding + extend_image_doubles + padding
            )

    fig, axes = plot_dsprites_images(plot_images, ncols=ncols)
    if title is not None:
        fig.suptitle(title, y=1.04)
    
    x_left = axes[0, 0].get_position().x0
    x_right = axes[-1, -1].get_position().x1
    x_ext = (x_right - x_left) / 30
    for r, row_start_ax in enumerate(axes[:, 0]):
        ylabel = "Images" if not r % 2 else doubles_str
        row_start_ax.set_ylabel(ylabel)

        if r != 0 and not r % 2:
            top_ax_y = axes[r - 1, 0].get_position().y0
            bot_ax_y = axes[r, 0].get_position().y1
            y = np.mean([bot_ax_y, top_ax_y])

            line = plt.Line2D(
                [x_left - x_ext, x_right + x_ext], [y, y], 
                transform=fig.transFigure, color="black"
                )
            fig.add_artist(line)
    
    return fig, axes


def plot_RSMs(rsms, titles=None):
    """
    plot_RSMs(rsms)

    Plots representational similarity matrices.

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

    fig, axes = plt.subplots(
        ncols=ncols, figsize=[ncols * wid, wid], squeeze=False
        )
    fig.suptitle("Representational Similarity Matrices (RSMs)", y=1.05)

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

