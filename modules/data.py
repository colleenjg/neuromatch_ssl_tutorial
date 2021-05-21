import os

import numpy as np
from PIL import Image
import torch
import torchvision


DEFAULT_DATASET_NPZ_PATH = os.path.join("..", "dsprites", "dsprites_subset.npz")


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
        """

        self.dataset_path = dataset_path
        self.npz = np.load(self.dataset_path, allow_pickle=True, encoding="latin1")

        self.images = self.npz["imgs"][()]
        self.latent_classes = self.npz["latents_classes"][()]
        
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
        - latent_class_values (dict): possible latent class 
            values for each latent class, organized in 1D numpy arrays, 
            under latent class name keys. 
        - num_latent_class_values (1D np array): number of theoretically 
            possible values per latent, ordered as latent class names.
        - title (str): dataset title
        """

        metadata = self.npz["metadata"][()]

        self.date = metadata["date"]
        self.description = metadata["description"]
        self.version = metadata["version"]
        self.latent_class_names = metadata["latents_names"]
        self.latent_class_values = metadata["latents_possible_values"]
        self.num_latent_class_values = metadata["latents_sizes"]
        self.title = metadata["title"]
        

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

        if not isinstance(latent_class_names, list):
            latent_class_names = [latent_class_names]       
        
        latent_name_idxs = []
        for latent_class_name in latent_class_names:
            if latent_class_name not in self.latent_class_names:
                latent_names_str = ", ".join(self.latent_class_names)
                raise ValueError(
                    f"{latent_class_name} not recognized as a latent name. "
                    f"Must be in: {latent_names_str}."
                    )
            latent_name_idxs.append(self.latent_class_names.index(latent_class_name)) 

        return latent_name_idxs  


    def get_latent_values(self, class_indices=None, latent_class_names=None):
        """
        self.get_latent_values()

        Returns latent class values.

        Optional args:
        - class_indices (array-like): indices for which to return latent 
            class values. Order is preserved. If None, all are returned 
            (default: None).
        - latent_class_names (str or list): name(s) of latent class(es) 
            for which to return latent class values. Order is preserved. 
            If None, values for all latents are returned. (default: None)
        
        Returns:
        - (2D np array): array of latent class values (img x latent class)
        """

        if class_indices is not None:
            class_indices = np.asarray(class_indices)

        latent_class_name_idxs = self.get_latent_name_idxs(latent_class_names)

        return self.latent_class_values[class_indices][:, latent_class_name_idxs]



class CustomTorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, torchvision_transforms=None, rgb_expand=False, resize=None, 
                 simclr=False):
        """
        Initialized a custom Torch dataset.

        Required args:
        - X (2 or 3D np array): image array (channels (optional) x height x width).
        - y (1D np array): targets

        Optional args:
        - torchvision_transforms (torchvision.transforms): torchvision transforms to apply to X.
            (default: None)
        - rgb_expand (bool): if True, X is expanded to include 3 identical channels. Applied 
            after any torchvision_tranforms. (default: False)
        - resize (None or int): if not None, should be an int, namely the size to which X is 
            expanded along its height and width. (default: None)
        - simclr (bool): if True, specific SimCLR transformations are applied. Overrides any other 
            transforms. See self._apply_simclr_transforms(). (default: False)
        """

        self.X = X
        self.y = y
        
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same length.")
        
        if len(self.X.shape) not in [2, 3]:
            raise ValueError("Expect X to have 2 or 3 dimensions.")

        self.simclr = simclr
        if self.simclr:
            rgb_expand = False
            resize = None
            torchvision_transforms = False

            from simclr.modules.transformations import TransformsSimCLR
            if self.simclr == "test":
                self.simclr_transforms = TransformsSimCLR(size=224).test_transform
            else:
                self.simclr_transforms = TransformsSimCLR(size=224).train_transform

        self.torchvision_transforms = torchvision_transforms

        self.rgb_expand = rgb_expand
        if self.rgb_expand and len(X.shape) != 2:
            raise ValueError("If rgb_expand is True, X should have 2 dimensions.")
        
        self.resize = resize
        if self.resize is not None:
            self.resize_transform = torchvision.transforms.Resize(size=self.resize)
        
        self.num_samples = len(self.X)


    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        
        if self.torchvision_transforms:
            X = self.torchvision_transforms(X)
        if self.simclr:
            X = self._apply_simclr_transforms(X)
        else:
            X = torch.toTensor(X)

        if self.resize is not None:
            X = self.resize_transform(X)
        if self.rgb_expand:
            X = self.X.unsqueeze(dim=-3).expand(3, -1, -1)

        y = torch.toTensor(y)

        return X, y


    def _apply_simclr_transforms(self, X):
        """
        Applies SimCLR transformations for use with the SimCLR implementation available 
        here: https://github.com/Spijkervet/SimCLR

        Required args:
        - X (2 or 3D np array): image array (height x width x channels (optional)). 
            All values expected to be between 0 and 1.
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

        X = self.simclr_transforms(X)

        return X

