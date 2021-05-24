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

