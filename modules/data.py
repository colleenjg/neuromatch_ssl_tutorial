import os
import warnings

import numpy as np
import torch
from torch import nn
import torchvision

from . import plot_util


DEFAULT_DATASET_NPZ_PATH = os.path.join("dsprites", "dsprites_subset.npz")


def get_biased_indices(dataset, indices, bias="shape_posX", control=False, 
                       randst=None):
    """
    get_biased_indices(dataset, indices)

    Returns indices after removing those rejected given the requested bias. 
    For example, if the bias is 'heart_right', the indices of any images where 
    the heart is on the right are removed.

    Required args:
    - dataset (torch dSprites dataset): dSprites torch dataset
    - indices (1D np array): dataset image indices

    Optional args:
    - bias (str): way to bias the dataset subset defined by the indices.
        'heart_left': only include hearts on the left
        'shape_posX': correlate shape to posX
        (default: "heart_left")
    - control (bool): if True, the same number of items are excluded, as 
        determined by the bias, but they are randomly selected. 
        (default: False)
    - randst (torch Generator or int): random state to use when splitting 
        dataset. (default: None)

    Returns
    - indices (1D np array): indices retained
    """

    if bias == "heart_left":
        shapes, pos_Xs = dataset.dSprites.get_latent_values(
            indices, latent_class_names=["shape", "posX"]
            ).T

        heart_value = dataset.dSprites.shape_name_to_value_map["heart"]
        exclude_bool = ((shapes == heart_value) * (pos_Xs > 0.5))
    
    elif bias in ["shape_posX", "shape_posX_spaced"]:
        shapes, posXs = dataset.dSprites.get_latent_values(
            indices, latent_class_names=["shape", "posX"]
            ).T
       
        exclude_bool = np.zeros_like(indices).astype(bool)
        shape_vals = dataset.dSprites.latent_class_values["shape"]
        posX_vals = np.sort(dataset.dSprites.latent_class_values["posX"])
        if bias == "shape_posX":
            posX_val_splits = np.array_split(posX_vals, len(shape_vals)) # unequal split allowed
        elif bias == "shape_posX_spaced":
            posX_val_edges = [[0, 0.3], [0.35, 0.65], [0.7, 1.0]]
            posX_val_splits = [[
                val for val in posX_vals if val >= edges[0] and val < edges[1]
                ] for edges in posX_val_edges]
        for shape_val, pos_valX_split in zip(shape_vals, posX_val_splits):
            exclude_bool += (
                (shapes == shape_val) * ~np.isin(posXs, pos_valX_split)
                )
    
    else:
        raise NotImplementedError(
            f"{bias} bias is not implemented. Only 'heart_left' and " 
            "'shape_posX' biases are currently implemented."
            )

    if control: # randomly permute the exclusion boolean
        if isinstance(randst, int):
            randst = torch.random.manual_seed(randst)
        exclude_bool = exclude_bool[
            torch.randperm(len(exclude_bool), generator=randst)
            ]

    indices = indices[~exclude_bool]

    return indices


def subsample_sampler(sampler, fraction_sample=1.0, randst=None):
    """
    subsample_sampler(sampler)

    Required args:
    - sampler (SubsetRandomSampler): dataset sampler

    Optional args:
    - fraction_sample (float): fraction of sampler indices to retain in 
        new sample.(default: 1.0)
    - randst (torch Generator or int): random state to use when subsampling. 
        (default: None)
    
    Returns:
    - sub_sampler (SubsetRandomSampler): subset dataset sampler (unseeded)
    """
    
    if 1 <= fraction_sample <= 0:
        raise ValueError(
            "fraction_sample must be between 0 and 1, inclusively, but "
            f"found {fraction_sample}."
            )

    subset_size = int(fraction_sample * len(sampler.indices))

    if isinstance(randst, int):
        randst = torch.random.manual_seed(randst)

    sampler_indices = sampler.indices[
        torch.randperm(len(sampler.indices), generator=randst)
        ]

    sub_sampler = torch.utils.data.SubsetRandomSampler(
        sampler_indices[: subset_size]
        )

    return sub_sampler


def train_test_split_idx(dataset, fraction_train=0.8, randst=None, 
                         train_bias=None, control=False):
    """
    train_test_split_idx(dataset)

    Splits dataset into train and test (or any other set of 2 complementary 
    subsets).

    Required args:
    - dataset (torch dSprites dataset): dSprites torch dataset

    Optional args:
    - fraction_train (prop): fraction of dataset to allocate to training set. 
        (default 0.8)
    - randst (torch Generator or int): random state to use when splitting 
        dataset. (default: None)
    - train_bias (str): type of bias to introduce into the training dataset, 
        after the split is done, e.g., 'heart_left' (only hearts on left are 
        included) or 'shape_posX' (shape and posX are 
        correlated) (default: None)
    - control (bool): if True, the same number of items are removed from the 
        training dataset as the train_bias would determine, but they are 
        randomly selected. (default: False)

    Returns:
    - train_sampler (SubsetRandomSampler): training dataset sampler (unseeded)
    - test_indices (SubsetRandomSampler): test dataset sampler (unseeded)
    """

    if not hasattr(dataset, "dSprites"):
        raise ValueError("Expected dataset to be of type "
            f"dSpritesTorchDataset, but found {type(dataset)}.")

    if 1 <= fraction_train <= 0:
        raise ValueError(
            "fraction_train must be between 0 and 1, inclusively, but "
            f"found {fraction_train}."
            )

    train_size = int(fraction_train * len(dataset))

    if isinstance(randst, int):
        randst = torch.random.manual_seed(randst)

    all_indices = torch.randperm(len(dataset), generator=randst)

    train_indices = all_indices[: train_size]
    if train_bias is not None:
        if hasattr(dataset, "indices"):
            # implementing this just requires an extra indexing step
            raise NotImplementedError(
                "Training bias is implemented for full torch datasets only, "
                "not subsets."
                )
        train_indices = get_biased_indices(
            dataset, train_indices, bias=train_bias, control=control
            )
    test_indices = all_indices[train_size :]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    return train_sampler, test_sampler


class dSpritesDataset():
    
    def __init__(self, dataset_path=DEFAULT_DATASET_NPZ_PATH):
        """
        Initializes dSpritesDataset instance, sets basic attributes and 
        metadata attributes.

        Optional args:
        - dataset_path (str): path to dataset 
            (default: global variable DEFAULT_DATASET_NPZ_PATH)

        Attributes:
        - dataset_path (str): path to the dataset
        - npz (np.lib.bpyio.NpzFile): zipped numpy data file
        - num_images (int): number of images in the dataset
        """

        self.dataset_path = dataset_path
        self.npz = np.load(
            self.dataset_path, allow_pickle=True, encoding="latin1"
            )
        self._load_metadata()


    def __repr__(self):
        return f"dSprites dataset"

    
    @property
    def images(self):
        """
        Lazily load and returns all dataset images.

        - self._images: (3D np array): images (image x height x width)
        """

        if not hasattr(self, "_images"):
            self._images = self.npz["imgs"][()]
        return self._images


    @property
    def latent_classes(self):
        """
        Lazily load and returns latent classes for each dataset image.

        - self._latent_classes (3D np array): latent class values for each 
            image (image x latent)
        """

        if not hasattr(self, "_latent_classes"):
            self._latent_classes = self.npz["latents_classes"][()]

        return self._latent_classes

    @property
    def num_images(self):

        if not hasattr(self, "_num_images"):
            self._num_images = len(self.latent_classes)

        return self._num_images

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
        - value_to_shape_name_map (dict): mapping of shape values (1, 2, 3) to 
            shape names ("square", "oval", "heart") 
        - shape_name_to_value_map (dict): mapping of shape names 
            ("square", "oval", "heart") to shape values (1, 2, 3)
        """

        metadata = self.npz["metadata"][()]

        self.date = metadata["date"]
        self.description = metadata["description"]
        self.version = metadata["version"]
        self.latent_class_names = metadata["latents_names"]
        self.latent_class_values = metadata["latents_possible_values"]
        self.num_latent_class_values = metadata["latents_sizes"]
        self.title = metadata["title"]

        self.value_to_shape_name_map = {
            1: "square",
            2: "oval",
            3: "heart"
        }

        self.shape_name_to_value_map = {
            value: key for key, value in self.value_to_shape_name_map.items()
            }
        

    def _check_class_name(self, latent_class_name="shape"):
        """
        self._check_class_name()

        Raises an error if latent_class_name is not recognized.

        Optional args:
        - latent_class_name (str): name of latent class to check. 
            (default: "shape")
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
            latent_name_idxs.append(
                self.latent_class_names.index(latent_class_name)
                ) 

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


    def get_latent_values_from_classes(self, latent_classes, 
                                       latent_class_name="shape"):
        """
        self.get_latent_values_from_classes()

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
        

        num_classes = len(self.latent_class_values[latent_class_name])
        if (latent_classes >= num_classes).any():
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

        shape_names = [self.value_to_shape_name_map[int(value)] 
            for value in shape_values]

        return shape_names


    def show_images(self, indices=None, num_images=10, randst=None, 
                    annotations=None):
        """
        self.show_images()

        Plots dSprites images, as well as their latent values.
        Adapted from https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb

        Optional args:
        - indices (array-like): indices of images to plot. If None, they are 
            sampled randomly. (default: None)
        - num_images (int): number of images to sample and plot, if indices 
            is None. (default: 10)
        - randst (np.random.RandomState): seed or random state to use if 
            sampling images. If None, the global state is used.
            (default: None)
        - annotations (str): If not None, annotations are added to images, 
            e.g., 'posX_quadrants'. (default: None)
        """

        if indices is None:
            if num_images > self.num_images:
                raise ValueError("Cannot sample more images than the number "
                    f"of images in the dataset ({self.num_images}).")
            if randst is None:
                randst = np.random
            elif isinstance(randst, int):
                randst = np.random.RandomState(randst)
            indices = randst.choice(
                np.arange(self.num_images), num_images, replace=False
                )
        else:
            num_images = len(indices)

        imgs = self.images[indices]
        
        centers = None
        annotation_str = ""
        y = 1.04
        if annotations is not None:
            centers = self.get_latent_values(
                indices, latent_class_names=["posX", "posY"]
                )
            annotation_str = "\nwith annotations (red)"
            y = 1.1

        fig, axes = plot_util.plot_dsprites_images(
            imgs, annotations=annotations, centers=centers
            )
        ncols = axes.shape[1]
        axes = axes.flatten()

        # retrieve latent values and shape names
        latent_values = self.get_latent_values(indices)
        shape_names = self.get_shapes_from_values(latent_values[:, 0])

        fig.suptitle(
            (f"{num_images} images sampled from the dSprites "
            f"dataset{annotation_str}"), y=y
            )
        for ax_i, ax in enumerate(axes.flatten()):
            if ax_i < num_images:
                img_latent_values = [
                    f"{value:.2f}" for value in latent_values[ax_i
                    ]]
                img_latent_values[0] = \
                    f"{latent_values[ax_i, 0]} ({shape_names[ax_i]})"
                if not (ax_i % ncols):
                    title = "\n".join(
                        [f"{name}: {value}" for name, value in zip(
                            self.latent_class_names, img_latent_values)
                            ]
                        )
                else:
                    title = "\n".join(img_latent_values)
                ax.set_xlabel(title, fontsize="x-small")


class dSpritesTorchDataset(torch.utils.data.Dataset):
    def __init__(self, dSprites, target_latent="shape", 
                 torchvision_transforms=None, resize=None, rgb_expand=False, 
                 simclr=False, simclr_mode="train", simclr_transforms=None):
        """
        Initialized a custom Torch dataset for dSprites, and sets attributes.

        NOTE: Always check that transforms behave as expected (e.g., produce   
        outputs in expected range), as datatypes (e.g., torch vs numpy, 
        uint8 vs float32) can change the behaviours of certain transforms, 
        e.g. ToPILImage.

        Required args:
        - dSprites (dSpritesDataset): dSprites dataset

        Optional args:
        - target_latent (str): latent dimension to use as target. 
            (default: "shape")
        - torchvision_transforms (torchvision.transforms): torchvision 
            transforms to apply to X. (default: None)
        - resize (None or int): if not None, should be an int, namely the 
            size to which X is expanded along its height and width. 
            (default: None)
        - rgb_expand (bool): if True, X is expanded to include 3 identical 
            channels. Applied after any torchvision_tranforms. 
            (default: False)
        - simclr (bool or str): if True, SimCLR-specific transformations are 
            applied. (default: False)
        - simclr_mode (str): If not None, determines whether data is returned 
            in 'train' mode (with augmentations) or 'test' mode (no augmentations). 
            Ignored if simclr is False. 
            (default: 'train') 
        - simclr_transforms (torchvision.transforms): SimCLR-specific 
            transforms. If "spijk", then SimCLR transforms from (https://github.com/Spijkervet/SimCLR), 
            are ised. If None, default SimCLR transforms are applied. Ignored if 
            simclr is False. (default: None)

        Sets attributes:
        - X (2 or 3D np array): image array 
            (channels (optional) x height x width).
        - y (1D np array): targets

        ...
        """

        self.dSprites = dSprites
        self.target_latent = target_latent

        self.X = self.dSprites.images
        self.y = self.dSprites.get_latent_classes(
            latent_class_names=target_latent
            ).squeeze()
        self.num_classes = \
            len(self.dSprites.latent_class_values[self.target_latent])
        
        if len(self.X) != len(self.y):
            raise ValueError(
                "images and latent classes must have the same length, but "
                f"found {len(self.X)} and {len(self.y)}, respectively."
                )
        
        if len(self.X.shape) not in [3, 4]:
            raise ValueError("images should have 3 or 4 dimensions, but "
                f"found {len(self.X.shape)}.")

        self.simclr = simclr
        self.simclr_mode = None
        self.simclr_transforms = None
        if self.simclr:
            self.simclr_mode = simclr_mode
            self.spijk = (simclr_transforms == "spijk")
            if self.simclr_mode not in ["train", "test"]:
                raise ValueError("simclr_mode must be 'train' or 'test', but "
                    f"found {self.simclr_mode}.")

            if self.spijk:
                torchvision_transforms = False
                if len(self.X[0].shape) == 2:
                    rgb_expand = True

                from simclr.modules.transformations import TransformsSimCLR
                if self.simclr_mode == "train":
                    self.simclr_transforms = \
                        TransformsSimCLR(size=224).train_transform
                else:
                    self.simclr_transforms = \
                        TransformsSimCLR(size=224).test_transform

            else:
                if self.simclr_mode == "train":
                    self.simclr_transforms = simclr_transforms
                    if self.simclr_transforms is None:
                        self.simclr_transforms = \
                            torchvision.transforms.RandomAffine(
                                degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)
                            )
                else:
                    self.simclr_transforms = None

        self.torchvision_transforms = torchvision_transforms
        
        self.resize = resize
        if self.resize is not None:
            self.resize_transform = \
                torchvision.transforms.Resize(size=self.resize)

        self.rgb_expand = rgb_expand
        if self.rgb_expand and len(self.X[0].shape) != 2:
            raise ValueError(
                "If rgb_expand is True, X should have 2 dimensions, but it"
                f" has {len(self.X[0].shape)} dimensions."
                )

        self._ch_expand = False
        if len(self.X[0].shape) == 2 and not self.rgb_expand:
            self._ch_expand = True

        self.num_samples = len(self.X)


    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):
        X = self.X[idx].astype(np.float32)
        y = self.y[idx]

        if self.rgb_expand:
            X = np.repeat(np.expand_dims(X, axis=-3), 3, axis=-3)

        if self._ch_expand:
            X = np.expand_dims(X, axis=-3)

        X = torch.tensor(X)


        if self.simclr and self.spijk:
            X = self._preprocess_simclr_spijk(X)
        else:
            if self.resize is not None:
                X = self.resize_transform(X)

            if self.torchvision_transforms is not None:
                X = self.torchvision_transforms()(X)

        y = torch.tensor(y)

        if self.simclr:
            if self.simclr_transforms is None: # e.g. in test mode
                X_aug1, X_aug2 = X, X
            else:
                X_aug1 = self.simclr_transforms(X)
                X_aug2 = self.simclr_transforms(X) 
            return (X_aug1, X_aug2, y, idx)
        else:
            return (X, y, idx)


    def _preprocess_simclr_spijk(self, X):
        """
        self._preprocess_simclr_spijk(X)
        
        Preprocess X for SimCLR transformations of the SimCLR implementation 
        available here: https://github.com/Spijkervet/SimCLR

        Required args:
        - X (2 or 3D np array): image array 
            (height x width x channels (optional)). 
            All values expected to be between 0 and 1.
        
        Returns:
        - X (3 or 4D np array): image array 
                                ((images) x height x width x channels).
        """

        if X.max() > 1 or X.min() < 0:
            raise NotImplementedError(
                "Expected X to be between 0 and 1 for SimCLR transform."
                )

        if len(X.shape) == 4:
            raise NotImplementedError(
                "Slicing dataset with multiple index values at once not "
                "supported, due to use of PIL torchvision transforms."
                )
        
        # input must be torch Tensor to be correctly interpreted
        X = torchvision.transforms.ToPILImage(mode="RGB")(X)

        return X


    def show_images(self, indices=None, num_images=10, ncols=5, randst=None, 
                    annotations=None):
        """
        self.show_images()

        Plots dSprites images, or their augmentations if applicable.

        Optional args:
        - indices (array-like): indices of images to plot. If None, they are 
            sampled randomly. (default: None)
        - num_images (int): number of images to sample and plot, if indices is 
            None. (default: 10)
        - ncols (int): number of columns to plot. (default: 5)
        - randst (np.random.RandomState): seed or random state to use if 
            sampling images. If None, the global state is used. (Does not 
            control SimCLR transformations.) (default: None)
        - annotations (str): If not None, annotations are added to images, 
            e.g., 'posX_quadrants'. (default: None)
        """

        if indices is None:
            if num_images > self.num_samples:
                raise ValueError("Cannot sample more images than the number "
                    f"of images in the dataset ({self.num_samples}).")
            if randst is None:
                randst = np.random
            elif isinstance(randst, int):
                randst = np.random.RandomState(randst)
            indices = randst.choice(
                np.arange(self.num_samples), num_images, replace=False
                )
        else:
            num_images = len(indices)

        centers = None
        if annotations is not None:
            if self.simclr and self.simclr_mode == "train":
                # all data is augmented, so centers cannot be identified
                centers = None
            else:
                centers = self.dSprites.get_latent_values(
                    indices, latent_class_names=["posX", "posY"]
                    ).tolist()

        Xs, X_augs1, X_augs2 = [], [], []
        for idx in indices:
            if self.simclr:
                X_aug1, X_aug2, _, _ = self[idx]
                X_augs1.append(X_aug1.numpy())
                X_augs2.append(X_aug2.numpy())
            else:
                X, _, _ = self[idx]
                Xs.append(X.numpy())

        
        if self.simclr:
            title = f"{num_images} pairs of dataset image augmentations"
            fig, _ = plot_util.plot_dsprite_image_doubles(
                X_augs1, X_augs2, ["Augm. 1", "Augm. 2"], ncols=ncols, annotations=annotations, 
                centers=[centers, None]
                )
        else:
            title = f"{num_images} dataset images"
            fig, _ = plot_util.plot_dsprites_images(
                Xs, ncols=ncols, annotations=annotations, centers=centers
                )
        
        y = 1.04
        if annotations is not None:
            title = f"{title}\nwith annotations (red)"
            y = 1.1

        fig.suptitle(title, y=1.04)


def calculate_torch_RSM(features, features_comp=None, stack=False, 
                        mem_thr=1e5):
    """
    calculate_torch_RSM(features)

    Calculates representational similarity matrix (RSM) between two feature 
    matrices using pairwise cosine similarity. 
    Uses torch.nn.functional.cosine_similarity()

    Required args:
    - features (2D torch Tensor): feature matrix (items x features)

    Optional args
    - features_comp (2D torch Tensor): second feature matrix 
        (items x features). If None, features is compared to itself. 
        (default: None)
    - stack (bool): if True, feature and features_comp are first stacked 
        along the items dimension, and the resulting matrix is compared to 
        itself. (default: False)
    - mem_thr (num): limit of features size at which RSM is calculated in 
        blocks to avoid out-of-memory errors. (default: 5e5) 

    Returns:
    - rsm (2D torch Tensor): similarity matrix 
        (nbr features items x nbr features_comp items)
    """

    if features_comp is None:
        if stack:
            raise ValueError(
                "stack cannot be set to True if features_comp is None."
                )
        features_comp = features
    else:
        if features.shape != features_comp.shape:
            raise ValueError(
                "features and features_comp should have the same shape, but "
                f"found shapes {features.shape} and {features_comp.shape} "
                "respectively."
                )
        features = torch.cat((features, features_comp), dim=0)
        features_comp = features


    n_blocks = int(np.ceil(np.product(features.shape) / mem_thr))
    n = int(np.ceil(len(features) / n_blocks))

    if n_blocks > 1:
        warnings.warn(f"Calculating RSM in {n_blocks} blocks to avoid "
            "out-of-memory errors.")

    rsm = torch.empty(len(features), len(features))

    for i in range(n_blocks):
        i_slice = slice(i * n, (i + 1) * n)
        for j in range(n_blocks):
            j_slice = slice(j * n, (j + 1) * n)
            rsm[i_slice, j_slice] = \
                nn.functional.cosine_similarity(
                torch.flatten(features[i_slice], start_dim=1).unsqueeze(1), 
                torch.flatten(features_comp[j_slice], start_dim=1).unsqueeze(0), 
                dim=2
                )
    
    return rsm


def calculate_numpy_RSM(features, features_comp=None, stack=False, 
                        centered=False):
    """
    calculate_numpy_RSM(features)

    Calculates representational similarity matrix (RSM) between two feature 
    matrices using pairwise cosine similarity. If centered is True, this 
    calculation is equivalent to pairwise Pearson correlations. Uses numpy.

    Required args:
    - features (2D np array): feature matrix (items x features)

    Optional args
    - features_comp (2D np array): second feature matrix (items x features). 
        If None, features is compared to itself. (default: None)
    - stack (bool): if True, feature and features_comp are first stacked 
        along the items dimension, and the resulting matrix is compared to 
        itself. (default: False)
    - centered (bool): if True, the mean across features is first subtracted 
        for each item. (default: False)  

    Returns:
    - rsm (2D np array): similarity matrix 
        (nbr features items x nbr features_comp items)
    """

    if features_comp is None:
        if stack:
            raise ValueError(
                "stack cannot be set to True if features_comp is None."
                )
        features_comp = features
    else:
        if features.shape != features_comp.shape:
            raise ValueError(
                "features and features_comp should have the same shape, but "
                f"found shapes {features.shape} and {features_comp.shape} "
                "respectively."
                )
        features = np.concatenate((features, features_comp), axis=0)
        features_comp = features

    norm_features, norms = [], []
    for _features in [features, features_comp]:
        _features = _features.reshape(len(_features), -1) # flatten
        
        if centered:
            _features -= np.mean(_features, axis=1, keepdims=True)

        # calculate L2 norms
        _norms = np.linalg.norm(_features, axis=1, keepdims=True) 

        norm_features.append(_features)
        norms.append(_norms)

    norms = np.maximum(np.dot(norms[0], norms[1].T), 1e-8) # raise to tolerance

    rsm = np.dot(norm_features[0], norm_features[1].T) / norms

    return rsm


def plot_dsprites_RSMs(dataset, rsms, target_class_values, titles=None, 
                       sorting_latent="shape"):
    """
    plot_dsprites_RSMs(dataset, rsms, target_class_values)

    Plots representational similarity matrices for dSprites data.

    Required args:
    - dataset (dSpritesDataset): dSprites dataset
    - rsms (list): list of 2D RSMs arrays.
    - target_class_values (list): list of target class values for each 
        element in the corresponding RSM. 

    Optional args:
    - titles (list): title for each RSM. (default: None)
    - sorting_latent (str): name of latent class/feature to sort rows 
        and columns by. (default: "shape")
    """

    if isinstance(rsms, list):
        if len(rsms) != len(target_class_values):
            raise ValueError(
                f"Must pass as many target_class_values as rsms ({len(rsms)})."
                ) 
        if not isinstance(titles, list) or len(titles) != len(rsms):
            raise ValueError(
                f"Must pass as many titles as rsms ({len(rsms)})."
                )
    
    else: # place in lists
        rsms = [rsms]
        target_class_values = [target_class_values]
        titles = [titles]
    
    for r, rsm_target_class_values in enumerate(target_class_values):
        if len(rsm_target_class_values) != len(rsms[r]):
            raise ValueError(
                "Must provide as many target_class_values as RSM rows/cols "
                f"({len(rsms[r])})."
                )
        sorter = np.argsort(rsm_target_class_values)
        target_class_values[r] = rsm_target_class_values[sorter]
        rsms[r] = rsms[r][sorter][:, sorter]

    _, axes = plot_util.plot_RSMs(rsms, titles)

    dataset._check_class_name(sorting_latent)

    for subax, sub_targ_class_vals in zip(axes.flatten(), target_class_values):
        
        # check that target classes are sorted, and collect unique values 
        # and where they start
        target_change_idxs = np.insert(
            np.where(np.diff(sub_targ_class_vals))[0] + 1, 
            0, 0)
        unique_values = [sub_targ_class_vals[i] for i in target_change_idxs]
        if sorting_latent == "shape":
            unique_values = dataset.get_shapes_from_values(unique_values)
        elif sorting_latent == "scale":
            unique_values = [f"{value:.1f}" for value in unique_values]
        
        # place major ticks at class boundaries and class labels between
        sorting_latent_str = sorting_latent
        if sorting_latent in ["shape", "scale"]:
            edge_ticks = np.append(
                target_change_idxs, len(sub_targ_class_vals)
                )
            label_ticks = target_change_idxs + np.diff(edge_ticks) / 2

            for axis, rotation in zip(
                [subax.xaxis, subax.yaxis], ["horizontal", "vertical"]
                ):
                if rotation == "horizontal":
                    kwargs = {"ha": "center"}
                else:
                    kwargs = {"va": "center"}

                axis.set_ticks(edge_ticks.tolist())
                axis.set_tick_params(width=2, length=10, which="major")
                axis.set_ticklabels("", minor=False)

                axis.set_ticks(label_ticks, minor=True)
                axis.set_tick_params(length=0, which="minor")
                axis.set_ticklabels(
                    unique_values, minor=True, fontsize=14, rotation=rotation, 
                    **kwargs
                    )

        else:
            if sorting_latent == "orientation":
                sorting_latent_str = f"{sorting_latent} (in radians)"
                nticks = 9
            elif sorting_latent in ["posX", "posY"]:
                nticks = 11

            possible_values = dataset.latent_class_values[sorting_latent]            
            min_val = possible_values.min()
            max_val = possible_values.max()

            ticks = np.linspace(0, len(sub_targ_class_vals), nticks)
            ticklabels = np.linspace(min_val, max_val, nticks)
            ticklabels = [f"{ticklabel:.1f}" for ticklabel in ticklabels]      

            for axis in [subax.xaxis, subax.yaxis]:
                axis.set_ticks(ticks)
                axis.set_ticklabels(ticklabels)

        subax.set_xlabel(sorting_latent_str, labelpad=10)

