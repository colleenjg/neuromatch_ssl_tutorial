import os

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision

from . import plot_util


DEFAULT_DATASET_NPZ_PATH = os.path.join("dsprites", "dsprites_subset.npz")


class dSpritesDataset():
    
    def __init__(self, dataset_path=DEFAULT_DATASET_NPZ_PATH):
        """
        Initializes dSpritesDataset instance, sets basic attributes and 
        metadata attributes.

        Attributes:
        - dataset_path (str): path to the dataset
        - npz (np.lib.bpyio.NpzFile): zipped numpy data file
        - images (3D np array): images (image x height x width)
        - latent_classes (2D np array): latent class values for each image (image x latent)
        - num_images (int): number of images in the dataset
        """

        self.dataset_path = dataset_path
        self.npz = np.load(self.dataset_path, allow_pickle=True, encoding="latin1")

        self.images = self.npz["imgs"][()]
        self.latent_classes = self.npz["latents_classes"][()]
        self.num_images = len(self.images)
        
        self._load_metadata()


    def __repr__(self):
        return f"dSprites dataset"


    def _load_metadata(self):
        """
        self._load_metadata()

        Sets metadata attributes.

        Attributes:
        - date (str): date the dataset was created
        - description (str): dataset description
        - version (str): version number
        - latent_class_names (tuple): ordered latent class names
        - latent_class_values (dict): latent values for each latent class, 
            organized in 1D numpy arrays, under latent class name keys. 
        - num_latent_class_values (1D np array): number of theoretically 
            possible values per latent, ordered as latent class names.
        - title (str): dataset title
        - shape_map (dict): mapping of shape values (1, 2, 3) to shape names 
            ("square", "oval", "heart") 
        """

        metadata = self.npz["metadata"][()]

        self.date = metadata["date"]
        self.description = metadata["description"]
        self.version = metadata["version"]
        self.latent_class_names = metadata["latents_names"]
        self.latent_class_values = metadata["latents_possible_values"]
        self.num_latent_class_values = metadata["latents_sizes"]
        self.title = metadata["title"]

        self.shape_name_map = {
            1: "square",
            2: "oval",
            3: "heart"
        }
        

    def _check_class_name(self, latent_class_name="shape"):
        """
        self._check_class_name()

        Raises an error if latent_class_name is not recognized.

        Optional args:
        - latent_class_name (str): name of latent class to check. (default: "shape")
        """
        if latent_class_name not in self.latent_class_names:
            latent_names_str = ", ".join(self.latent_class_names)
            raise ValueError(
                f"{latent_class_name} not recognized as a latent class name. "
                f"Must be in: {latent_names_str}."
                )


    def get_latent_name_idxs(self, latent_class_names=None):
        """
        self.get_latent_name_idxs()

        Returns indices for latent class names.

        Optional args:
        - latent_class_names (str or list): name(s) of latent class(es) for 
            which to return indices. Order is preserved. If None, indices 
            for all latents are returned. (default: None)
        
        Returns:
        - (list): list of latent class indices
        """

        if latent_class_names is None:
            return np.arange(len(self.latent_class_names))

        if not isinstance(latent_class_names, (list, tuple)):
            latent_class_names = [latent_class_names]       
        
        latent_name_idxs = []
        for latent_class_name in latent_class_names:
            self._check_class_name(latent_class_name)
            latent_name_idxs.append(self.latent_class_names.index(latent_class_name)) 

        return latent_name_idxs  


    def get_latent_classes(self, indices=None, latent_class_names=None):
        """
        self.get_latent_classes()

        Returns latent classes for each image.

        Optional args:
        - indices (array-like): image indices for which to return latent 
            class values. Order is preserved. If None, all are returned 
            (default: None).
        - latent_class_names (str or list): name(s) of latent class(es) 
            for which to return latent class values. Order is preserved. 
            If None, values for all latents are returned. (default: None)
        
        Returns:
        - (2D np array): array of latent classes (img x latent class)
        """

        if indices is not None:
            indices = np.asarray(indices)
        else:
            indices = slice(None)

        latent_class_name_idxs = self.get_latent_name_idxs(latent_class_names)

        return self.latent_classes[indices][:, latent_class_name_idxs]


    def get_latent_values_from_classes(self, latent_classes, latent_class_name="shape"):
        """
        self.get_latent_values()

        Returns latent class values for each image.

        Required args:
        - latent_classes (1D np array): array of class values for each image
        
        Optional args:
        - latent_class_name (str): name of latent class for which to return 
            latent class values. (default: "shape")
        
        Returns:
        - (2D np array): array of latent class values (img x latent class)
        """

        self._check_class_name(latent_class_name)

        latent_classes = np.asarray(latent_classes)

        if (latent_classes < 0).any():
            raise ValueError("Classes cannot be below 0.")
        if (latent_classes > len(self.latent_class_values[latent_class_name])).any():
            raise ValueError("Classes cannot exceed the number of class "
                "values for the latent class.")

        return self.latent_class_values[latent_class_name][latent_classes]


    def get_latent_values(self, indices=None, latent_class_names=None):
        """
        self.get_latent_values()

        Returns latent class values for each image.

        Optional args:
        - class_indices (array-like): image indices for which to return 
            latent class values. Order is preserved. If None, all are 
            returned (default: None).
        - latent_class_names (str or list): name(s) of latent class(es) 
            for which to return latent class values. Order is preserved. 
            If None, values for all latents are returned. (default: None)
        
        Returns:
        - latent_values (2D np array): array of latent class values 
            (img x latent class)
        """

        latent_classes = self.get_latent_classes(indices, latent_class_names)

        if latent_class_names is None:
            latent_class_names = self.latent_class_names

        if not isinstance(latent_class_names, (list, tuple)):
            latent_class_names = [latent_class_names]
        
        latent_values = np.empty_like(latent_classes).astype(float)
        for l, latent_class_name in enumerate(latent_class_names):
            latent_values[:, l] = self.get_latent_values_from_classes(
                latent_classes[:, l], latent_class_name
                )
            
        return latent_values


    def get_shapes_from_values(self, shape_values):
        """
        self.get_shapes_from_values()

        Returns shape name for each numerical shape value.

        Required args:
        - shape_values (array-like): numerical shape values (default: None).
        
        Returns:
        - shape_names (list): shape name for each numerical shape value
        """

        if set(shape_values) - set([1, 2, 3]):
            raise ValueError("Numerical shape values include only 1, 2 and 3.")

        shape_names = [self.shape_name_map[int(value)] for value in shape_values]

        return shape_names


    def show_images(self, indices=None, num_images=10, randst=None):
        """
        self.show_images()

        Plots dSprites images, as well as their latent values.
        Adapted from https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb

        Optional args:
        - indices (array-like): indices of images to plot. If None, they are sampled randomly.
            (default: None)
        - num_images (int): number of images to sample and plot, if indices is None.
            (default: 10)
        - randst (np.random.RandomState): seed or random state to use if sampling images.
            If None, the global state is used.
            (default: None)
        """

        if indices is None:
            if num_images > self.num_images:
                raise ValueError("Cannot sample more images than the number of images "
                    f"in the dataset ({self.num_images}).")
            if randst is None:
                randst = np.random
            elif isinstance(randst, int):
                randst = np.random.RandomState(randst)
            indices = randst.choice(np.arange(self.num_images), num_images, replace=False)
        else:
            num_images = len(indices)

        imgs = self.images[indices]
        fig, axes = plot_util.plot_dsprites_images(imgs)
        ncols = axes.shape[1]
        axes = axes.flatten()

        # retrieve latent values and shape names
        latent_values = self.get_latent_values(indices)
        shape_names = self.get_shapes_from_values(latent_values[:, 0])

        fig.suptitle(f"{num_images} images sampled from the dSprites dataset", y=1.04)
        for ax_i, ax in enumerate(axes.flatten()):
            if ax_i < num_images:
                img_latent_values = [f"{value:.2f}" for value in latent_values[ax_i]]
                img_latent_values[0] = f"{latent_values[ax_i, 0]} ({shape_names[ax_i]})"
                if not (ax_i % ncols):
                    title = "\n".join([f"{name}: {value}" for name, value in zip(
                        self.latent_class_names, img_latent_values)])
                else:
                    title = "\n".join(img_latent_values)
                ax.set_xlabel(title, fontsize="x-small")


class dSpritesTorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, torchvision_transforms=None, resize=None, rgb_expand=False, 
                 simclr=False, spijk=None, simclr_transforms=None):
        """
        Initialized a custom Torch dataset for dSprites.

        Required args:
        - X (2 or 3D np array): image array (channels (optional) x height x width).
        - y (1D np array): targets

        Optional args:
        - torchvision_transforms (torchvision.transforms): torchvision transforms to apply to X.
            (default: None)
        - resize (None or int): if not None, should be an int, namely the size to which X is 
            expanded along its height and width. (default: None)
        - rgb_expand (bool): if True, X is expanded to include 3 identical channels. Applied 
            after any torchvision_tranforms. (default: False)
        - simclr (bool or str): if True, SimCLR-specific transformations are applied. (default: False)
        - spijk (str): If not None, the SimCLR transforms are drawn from this implementation 
            (https://github.com/Spijkervet/SimCLR), using either "train" or "test" transforms, as 
            specified. All other transforms are overridden. Ignored if simclr is False. (default: None) 
        - simclr_transforms (torchvision.transforms): SimCLR-specific transforms. Used only if simclr 
            is True, and spikj is None. If None, default SimCLR transforms are applied. (default: None)
        """

        self.X = X
        self.y = y.squeeze()
        
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same length.")
        
        if len(self.X.shape) not in [2, 3]:
            raise ValueError("X should have 2 or 3 dimensions.")

        self.simclr = simclr
        if self.simclr:
            self.spijk = spijk
            if self.spijk is not None:
                rgb_expand = False
                resize = None
                torchvision_transforms = False
                
                if self.spijk not in ["train", "test"]:
                    raise ValueError("spijk must be 'train' or 'test'.")
                from simclr.modules.transformations import TransformsSimCLR
                if self.spijk == "train":
                    self.simclr_transforms = TransformsSimCLR(size=224).train_transform
                else:
                    self.simclr_transforms = TransformsSimCLR(size=224).test_transform

            else:
                self.simclr_transforms = simclr_transforms
                if self.simclr_transforms is None:
                    self.simclr_transforms = torchvision.transforms.RandomAffine(
                        degrees=180, translate=(0.2, 0.2)
                    ) # trying 180

        self.torchvision_transforms = torchvision_transforms
        
        self.resize = resize
        if self.resize is not None:
            self.resize_transform = torchvision.transforms.Resize(size=self.resize)

        self.rgb_expand = rgb_expand
        if self.rgb_expand and len(X.shape) != 2:
            raise ValueError("If rgb_expand is True, X should have 2 dimensions.")

        self.num_samples = len(self.X)


    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        if self.simclr == "spijk":
            X = self._preprocess_simclr_spijk(X)
        else:
            if self.torchvision_transforms:
                X = self.torchvision_transforms()(X)

            if self.resize is not None:
                X = self.resize_transform(X)

            if self.rgb_expand:
                X = np.repeat(np.expand_dims(X, axis=-3), 3, axis=0) 

        X = torchvision.transforms.ToTensor()(X)
        y = torch.tensor(y)

        if self.simclr:
            X_aug = self.simclr_transforms(X)
            return X, X_aug, y

        else:
            return X, y


    def _preprocess_simclr_spijk(self, X):
        """
        self._preprocess_simclr_spijk(X)
        
        Preprocess X for SimCLR transformations of the SimCLR implementation available 
        here: https://github.com/Spijkervet/SimCLR

        Required args:
        - X (2 or 3D np array): image array (height x width x channels (optional)). 
            All values expected to be between 0 and 1.
        
        Returns:
        - X (3D np array): image array (height x width x channels).
        """

        if len(X.shape) == 2:
            X = np.repeat(np.expand_dims(X, axis=-1), 3, axis=-1)    
        elif len(X.shape) == 3:
            X = np.transpose(X, [1, 2, 0]) # place channels last
        else:
            raise ValueError("Expected a 2 or 3-dimensional input for X for SimCLR transform.")
        
        if X.max() >= 1 or X.min() <= 0:
            raise NotImplementedError("Expected X to be between 0 and 1 for SimCLR transform.")

        X = Image.fromarray(np.uint8(X * 255)).convert("RGB")

        return X


    def show_images(self, indices=None, num_images=10, ncols=5, randst=None):
        """
        self.show_images()

        Plots dSprites images, as well as their augmentations if applicable.

        Optional args:
        - indices (array-like): indices of images to plot. If None, they are sampled randomly.
            (default: None)
        - num_images (int): number of images to sample and plot, if indices is None.
            (default: 10)
        - ncols (int): number of columns to plot. (default: 5)
        - randst (np.random.RandomState): seed or random state to use if sampling images.
            If None, the global state is used. (Does not control SimCLR transformations.)
            (default: None)
        """

        if indices is None:
            if num_images > self.num_samples:
                raise ValueError("Cannot sample more images than the number of images "
                    f"in the dataset ({self.num_samples}).")
            if randst is None:
                randst = np.random
            elif isinstance(randst, int):
                randst = np.random.RandomState(randst)
            indices = randst.choice(np.arange(self.num_samples), num_images, replace=False)
        else:
            num_images = len(indices)

        Xs = []
        if self.simclr:
            X_augs = []

        for idx in indices:
            if self.simclr:
                X, X_aug, _ = self[idx]
                X_augs.append(X_aug.numpy())
            else:
                X, _ = self[idx]
            Xs.append(X.numpy())
        
        if len(indices) and len(X.shape) == 3:
            Xs = np.mean(Xs, axis=1).tolist() # across channels
            if self.simclr:
                X_augs = np.mean(X_augs, axis=1).tolist()

        plot_Xs = Xs
        aug_str = ""
        if self.simclr:
            aug_str = " and augmentations"
            plot_Xs = []
            ncols = np.min([len(Xs), ncols])
            n_sets = int(np.ceil(len(Xs) / ncols))
            for i in range(n_sets):
                extend_Xs = Xs[i * ncols : (i + 1) * ncols]
                extend_X_augs = X_augs[i * ncols : (i + 1) * ncols]
                padding = [None] * (ncols - len(extend_Xs))

                plot_Xs.extend(extend_Xs + padding + extend_X_augs + padding)

        fig, axes = plot_util.plot_dsprites_images(plot_Xs, ncols=ncols)
        fig.suptitle(f"{num_images} dataset images{aug_str}", y=1.04)
        
        if self.simclr:
            x_left = axes[0, 0].get_position().x0
            x_right = axes[-1, -1].get_position().x1
            x_ext = (x_right - x_left) / 30
            for r, row_start_ax in enumerate(axes[:, 0]):
                ylabel = "Images" if not r % 2 else "Augm."
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

