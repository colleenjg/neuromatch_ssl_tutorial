import copy

from matplotlib import pyplot as plt
from matplotlib import colors as mplcol
import numpy as np


def add_annotations(image, annotations=None, center=None, color=None):
    """
    - annotations (str): If not None, annotations are added to images, 
        e.g., 'posX_quadrants'. (default: None)
    - centers (list): If not None, centers are provided to annotate the 
        images, in form [image_centers, image_double_centers], where 
        image_centers and image_double_centers are iterables. (default: None)    
    """
    
    image = copy.deepcopy(image)

    HEI, WID = 64, 64
    BUFFER = 16
    X_SPACING = 11
    N_QUADS = 3
    RADIUS = 2

    hei, wid = image.shape
    rel_hei = hei / HEI
    rel_wid = wid / WID

    x_buffer = int(np.around(rel_wid * BUFFER))
    y_buffer = int(np.around(rel_hei * BUFFER))

    if color is None:
        color = np.max(image) * 2

    if annotations is not None:
        if annotations not in ["pos", "posX_quadrants"]:
            raise ValueError(
                "If not None, annotations must be 'pos' or 'posX_quadrants'."
                )

        x_spacing = int(np.around(rel_wid * X_SPACING))
        
        # create dash square
        dash_len = 3
        hei_dash, wid_dash = [np.concatenate(
            [np.arange(i, v, dash_len * 2) for i in range(dash_len)]) 
            for v in [hei - y_buffer * 2, wid - x_buffer * 2]]

        image[y_buffer + hei_dash, x_buffer] = color
        image[y_buffer + hei_dash, wid - x_buffer] = color
        image[y_buffer, x_buffer + wid_dash] = color
        image[hei - y_buffer, x_buffer + wid_dash] = color

        # add dashed quadrant lines
        if annotations == "posX_quadrants":
            for n in range(1, N_QUADS):
                image[y_buffer + hei_dash, x_buffer + x_spacing * n] = color

        if center is not None:
            if len(center) != 2:
                raise ValueError(
                    "Expected 'centers' to have length 2, but found length "
                    f"{len(center)}."
                    )

            if np.max(center) > 1 or np.min(center) < 0:
                raise ValueError("Expected 'center' coordinates to be "
                    "between 0 and 1, inclusively.")    

            # obtain coordinates in pixels
            quadrant_width = wid - 2 * x_buffer
            quadrant_height = hei - 2 * x_buffer
            x_center = int(np.around((center[0] * quadrant_width + x_buffer)))
            y_center = int(np.around((center[1] * quadrant_height + y_buffer)))

            radius_adj = (np.mean([rel_hei, rel_wid]) * RADIUS)

            xx, yy = np.mgrid[: image.shape[0], : image.shape[1]]
            circle = (xx - x_center) ** 2 + (yy - y_center) ** 2
            image[np.where((circle < (radius_adj ** 2)).T)] = color


    return image


def plot_dsprites_images(images, ncols=5, title=None, annotations=None, 
                         centers=None):
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

    if annotations is None:
        color_list = ['black', 'white']
    else:
        color_list = ['black', 'white', 'red']

    if centers is None:
        if annotations is not None:
            centers = [None] * len(images)
    elif len(centers) != len(images):
            raise ValueError(
                "If providing centers, must provide as many as the number "
                "of images."
            )        
    
    cmap = mplcol.LinearSegmentedColormap.from_list(
        'dsprites_cmap', color_list, N=len(color_list))

    for ax_i, ax in enumerate(axes.flatten()):
        if images[ax_i] is not None and ax_i < num_images:
            image = images[ax_i]
            if annotations or centers:
                image = add_annotations(
                    image, annotations=annotations, center=centers[ax_i]
                    )
            ax.imshow(image, cmap=cmap, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    return fig, axes


def plot_dsprite_image_doubles(images, image_doubles, doubles_str, ncols=5, 
                               title=None, annotations=None, centers=None):
    """
    plot_dsprite_image_doubles(images, image_doubles, doubles_str)

    Plots dSprite images is sets of 2 rows.

    Required args:
    - images (list): list of images
    - image_doubles (list): list of image doubles (same length as images)
    - doubles_str (str or list): string that specified what the doubles are, 
        or list if specifying both images and image_doubles.

    Optional args:
    - ncols (int): number of columns. (default: 5)
    - title (str): plot title. If None, no title is included. (default: None)
    - annotations (str): If not None, annotations are added to images, 
        e.g., 'posX_quadrants'. (default: None)
    - centers (list): If not None, centers are provided to annotate the 
        images, in form [image_centers, image_double_centers], where 
        image_centers and image_double_centers are iterables. (default: None)         

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

    plot_centers = None
    if centers is not None:
        if len(centers) != 2:
            raise ValueError("centers must be of length 2 with center values "
                "(or None) for the images and image_doubles."
            )
        
        for s, sub_centers in enumerate(centers):
            if sub_centers is None:
                centers[s] = [None] * len(images)
            elif not isinstance(sub_centers, list):
                raise ValueError(
                    "Centers must comprise 2 lists: one for images and one "
                    "image_doubles (or None in either position)."
                    )
            elif len(sub_centers) != len(images):
                raise ValueError(
                    "Must provide as many values as images/images_double."
                    )

        plot_centers = []

    plot_images = []
    ncols = np.min([len(images), ncols])
    n_sets = int(np.ceil(len(images) / ncols))
    for i in range(n_sets):
        use_slice = slice(i * ncols, (i + 1) * ncols)
        extend_images = images[use_slice]
        extend_image_doubles = image_doubles[use_slice]
        
        padding = [None] * (ncols - len(extend_images))

        plot_images.extend(
            extend_images + padding + extend_image_doubles + padding
            )

        if plot_centers is not None:
            extend_image_centers = centers[0][use_slice]
            extend_image_double_centers = centers[1][use_slice]
            plot_centers.extend(
                extend_image_centers + padding + 
                extend_image_double_centers + padding
            )

    fig, axes = plot_dsprites_images(
        plot_images, ncols=ncols, annotations=annotations, centers=plot_centers
        )
    fig.tight_layout()
    if title is not None:
        fig.suptitle(title, y=1.04)

    images_str = "Images"
    if isinstance(doubles_str, list):
        if len(doubles_str) != 2:
            raise ValueError("If 'doubles_str' is a list, it must be of length 2.")
        images_str, doubles_str = doubles_str

    x_left = axes[0, 0].get_position().x0
    x_right = axes[-1, -1].get_position().x1
    x_ext = (x_right - x_left) / 30
    for r, row_start_ax in enumerate(axes[:, 0]):
        ylabel = images_str if not r % 2 else doubles_str
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

    min_val = np.min([rsm.min() for rsm in rsms] + [-1])
    max_val = np.max([rsm.max() for rsm in rsms] + [1])

    ncols = len(rsms)
    wid = 5

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

