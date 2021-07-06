import copy
from functools import partialmethod
import warnings

import numpy as np
import torch
from torch import nn
from tqdm.notebook import tqdm as tqdm
from matplotlib import pyplot as plt

from . import data, plot_util

DEFAULT_LABELLED_FRACTIONS = [0.05, 0.1, 0.2, 0.4, 0.75, 1.0]


def show_progress_bars(enable=True):
    """
    show_progress_bars()

    Enabled or disables tqdm progress bars.

    Optional args:
    - enabled (bool or str): progress bar setting ("reset" to previous)
    """

    if enable == "reset":
        if hasattr(tqdm, "_patch_prev_enable"):
            enable = tqdm._patch_prev_enable
        else:
            enable = True

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not(enable))
    tqdm._patch_prev_enable = not(enable)


def get_model_device(model):
    """
    get_model_device(model)

    Returns the device that the first parameters in a model are stored on.

    N.B.: Different components of a model can be stored on different devices. 
    Thisfunction does NOT check for this case, so it should only be used when 
    all model components are expected to be on the same device.

    Required args:
    - model (nn.Module): a torch model

    Returns:
    - first_param_device (str): device on which the first parameters of the 
        model are stored
    """
    

    if len(list(model.parameters())):
        first_param_device = next(model.parameters()).device
    else:
        first_param_device = "cpu" # default if the model has no parameters

    return first_param_device


class EncoderCore(nn.Module):
    def __init__(self, feat_size=84, input_dim=(1, 64, 64), vae=False):
        """
        Initializes the core encoder network.

        Optional args:
        - feat_size (int): size of the final features layer (default: 84)
        - input_dim (tuple): input image dimensions (channels, width, height) 
            (default: (1, 64, 64))
        - vae (bool): if True, a VAE encoder is initialized with a second 
            feature head for the log variances. (default: False)
        """

        super().__init__()

        self._vae = vae
        self._untrained = True

        # check input dimensions provided
        self.input_dim = tuple(input_dim)
        if len(self.input_dim) == 2:
            self.input_dim = (1, *input_dim)            
        elif len(self.input_dim) != 3:
            raise ValueError("input_dim should have length 2 (wid x hei) or "
                f"3 (ch x wid x hei), but has length ({len(self.input_dim)}).")
        self.input_ch = self.input_dim[0]

        # convolutional component of the feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_ch, out_channels=6, kernel_size=5, 
                stride=1
                ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(6, affine=False),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(16, affine=False)
        )

        # calculate size of the convolutional feature extractor output
        self.feat_extr_output_size = \
            self._get_feat_extr_output_size(self.input_dim)
        self.feat_size = feat_size

        # linear component of the feature extractor
        self.linear_projections = nn.Sequential(
            nn.Linear(self.feat_extr_output_size, 120),
            nn.ReLU(),
            nn.BatchNorm1d(120, affine=False),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.BatchNorm1d(84, affine=False),
        )

        self.linear_projections_output = nn.Sequential(
            nn.Linear(84, self.feat_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.feat_size, affine=False)
        )

        if self.vae:
            self.linear_projections_logvar = nn.Sequential(
                nn.Linear(84,self.feat_size),
                nn.ReLU(),
                nn.BatchNorm1d(self.feat_size,affine=False)
            )

    def _get_feat_extr_output_size(self, input_dim):
        dummy_tensor = torch.ones(1, *input_dim)
        reset_training = self.training
        self.eval()
        with torch.no_grad():   
            output_dim = self.feature_extractor(dummy_tensor).shape
        if reset_training:
            self.train()
        return np.product(output_dim)

    @property
    def vae(self):
        return self._vae

    @property
    def untrained(self):
        return self._untrained

    def forward(self, X):
        if self.untrained and self.training:
            self._untrained = False
        feats_extr = self.feature_extractor(X)
        feats_flat = torch.flatten(feats_extr, 1)
        feats_proj = self.linear_projections(feats_flat)
        feats = self.linear_projections_output(feats_proj)
        if self.vae:
            logvars = self. linear_projections_logvar(feats_proj)
            return feats, logvars
        return feats

    def get_features(self, X):
        with torch.no_grad():
            feats_extr = self.feature_extractor(X)
            feats_flat = torch.flatten(feats_extr, 1)
            feats_proj = self.linear_projections(feats_flat)
            feats = self.linear_projections_output(feats_proj)
        return feats
        

def train_classifier(encoder, dataset, train_sampler, test_sampler, 
                     num_epochs=10, fraction_of_labels=1.0, batch_size=1000, 
                     freeze_features=True, subset_seed=None, use_cuda=True, 
                     progress_bar=True, verbose=False):
    """
    train_classifier(encoder, dataset, train_sampler, test_sampler)

    Function to train a linear classifier to predict classes from features.
    
    Required args:
    - encoder (nn.Module): Encoder network instance for extracting features. 
        Should have method get_features(). If None, an Identity module is used.
    - dataset (dSpritesTorchDataset): dSprites torch dataset
    - train_sampler (SubsetRandomSampler): Training dataset sampler.
    - test_sampler (SubsetRandomSampler): Test dataset sampler.
    
    Optional args:
    - num_epochs (int): Number of epochs over which to train the classifier. 
        (default: 10)
    - fraction_of_labels (float): Fraction of the total number of available 
        labelled training data to use for training. (default: 1.0)
    - batch_size (int): Batch size. (default: 1000)
    - freeze_features (bool): If True, the feature encoder is frozen and only 
        the classifier is trained. If False, the encoder is also trained. 
        (default: True)
    - subset_seed (int): seed for selecting data subset, if applicable 
        (default: None)
    - use_cuda (bool): If True, cuda is used, if available. (default: True)
    - progress_bar (bool): If True, progress bars are enabled. (default: True)
    - verbose (bool): If True, classification accuracy is printed. 
        (default: False)

    Returns: 
    - classifier (nn.Linear): trained classification layer
    - loss_arr (list): training loss at each epoch
    - train_acc (float): final training accuracy
    - test_acc (float): final test accuracy
    """

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    if num_epochs is None:
        raise NotImplementedError(
            "Must set a number of epochs to an integer value."
            )

    if encoder is None:
        encoder = nn.Identity()
        encoder.get_features = encoder.forward
        encoder.untrained = True
        linear_input = dataset.dSprites.images[0].size
        if not freeze_features:
            raise ValueError(
                "freeze_features must be set to True if no encoder is passed"
                f", but is set to {freeze_features}."
                )
    else:
        linear_input = encoder.feat_size
    
    reset_encoder_device = get_model_device(encoder) # for later
    encoder.to(device)

    classifier = nn.Linear(linear_input, dataset.num_classes).to(device)

    if dataset.target_latent != "shape":
        warnings.warn(f"Training a logistic regression on "
            f"{dataset.target_latent} classification with "
            f"{dataset.num_classes} possible target classes.\nIf there is a "
            "meaningful linear relationship between the different classes, "
            "training a linear regression to predict latent values "
            "continuously would be advisable, instead of using a logistic "
            "regression.")

    if hasattr(dataset, "simclr") and dataset.simclr and not dataset.simclr_mode != "test":
        warnings.warn("Using a SimCLR dataset. Since the dataset returns 2 augmentations, "
            "the classifier will be trained on the first augmentation of each image.")

    # Define datasets and dataloaders
    train_subset_sampler = data.subsample_sampler(
        train_sampler, fraction_sample=fraction_of_labels, randst=subset_seed
        ) # obtain subset
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_subset_sampler
        )
    test_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler
        )

    # Define loss and optimizers
    train_parameters = classifier.parameters()
    if not freeze_features:
        train_parameters = list(train_parameters) + list(encoder.parameters())

    classification_optimizer = torch.optim.Adam(train_parameters, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        classification_optimizer, T_max=100
        )
    loss_fn = nn.CrossEntropyLoss()
    
    # Train classifier on training set
    classifier.train()
    reset_encoder_training = encoder.training
    if not freeze_features:
        encoder.train()
    elif not encoder.untrained: 
        encoder.eval() # otherwise untrained batch norm messes things up

    loss_arr = []
    for _ in tqdm(range(num_epochs), disable=not(progress_bar)):
        total_loss = 0
        num_total = 0
        for iter_data in train_dataloader:
            if dataset.simclr:
                X, _, y, _ = iter_data # ignore second X and indices
            else:
                X, y, _ = iter_data # ignore indices
            
            classification_optimizer.zero_grad()

            if freeze_features:
                features = encoder.get_features(X.to(device))
            else:
                features = encoder(X.to(device))

            predicted_y_logits = classifier(features.flatten(start_dim=1))
            loss = loss_fn(predicted_y_logits, y.to(device))
            loss.backward()
            classification_optimizer.step()

            total_loss += loss.item()
            num_total += y.size(0)

        loss_arr.append(total_loss / num_total)
        scheduler.step()
    
    # Calculate prediction accuracy on training and test sets
    classifier.eval()
    encoder.eval()

    accuracies = []
    for _, dataloader in enumerate((train_dataloader, test_dataloader)):
        num_correct = 0
        num_total = 0
        for iter_data in dataloader:
            if dataset.simclr:
                X, _, y, _ = iter_data # ignore second X and indices
            else:
                X, y, _ = iter_data # ignore indices

            with torch.no_grad():
                features = encoder.get_features(X.to(device))
                predicted_y_logits = classifier(features.flatten(start_dim=1))
            
            # identify predicted classes from logits
            _, predicted_y = torch.max(predicted_y_logits, 1)
            num_correct += (predicted_y.cpu() == y).sum()
            num_total += y.size(0)
            
        accuracy = (100 * num_correct.numpy()) / num_total
        accuracies.append(accuracy)

    train_acc, test_acc = accuracies

    # set final classifier state and reset original encoder state
    classifier.train()
    classifier.cpu()
    if reset_encoder_training:
        encoder.train()
    encoder.to(reset_encoder_device)

    if verbose:
        chance = 100 / dataset.num_classes
        if freeze_features:
            train_str = "classifier"
        else:
            train_str = "encoder and classifier"

        print(f"Network performance after {num_epochs} {train_str} training "
          f"epochs (chance: {chance:.2f}%):\n"
          f"    Training accuracy: {train_acc:.2f}%\n"
          f"    Testing accuracy: {test_acc:.2f}%")

    return classifier, loss_arr, train_acc, test_acc



def contrastive_loss(proj_feat1, proj_feat2, temperature=0.5, neg_pairs="all"):
    """
    contrastive_loss(proj_feat1, proj_feat2)

    Returns contrastive loss, given sets of projected features, with positive 
    pairs matched along the batch dimension.

    Required args:
    - proj_feat1 (2D torch Tensor): first set of projected features 
        (batch_size x feat_size)
    - proj_feat2 (2D torch Tensor): second set of projected features 
        (batch_size x feat_size)
      
    Optional args:
    - temperature (float): relaxation temperature. (default: 0.5)
    - neg_pairs (str or num): If "all", all available negative pairs are used
        for the loss calculation. Otherwise, only a certain number or 
        proportion of the negative pairs available in the batch, as specified 
        by the parameter, are randomly sampled and included in the 
        calculation, e.g. 5 for 5 examples or 0.05 for 5% of negative pairs. 
        (default: "all")

    Returns:
    - loss (float): mean contrastive loss
    """

    device = proj_feat1.device

    if len(proj_feat1) != len(proj_feat2):
        raise ValueError(f"Batch dimension of proj_feat1 ({len(proj_feat1)}) "
            f"and proj_feat2 ({len(proj_feat2)}) should be same")

    batch_size = len(proj_feat1) # N
    z1 = nn.functional.normalize(proj_feat1, dim=1)
    z2 = nn.functional.normalize(proj_feat2, dim=1)

    proj_features = torch.cat([z1, z2], dim=0) # 2N x projected feature dimension
    similarity_mat = nn.functional.cosine_similarity(
        proj_features.unsqueeze(1), proj_features.unsqueeze(0), dim=2
        ) # dim: 2N x 2N
    
    # initialize arrays to identify sets of positive and negative examples
    pos_sample_indicators = \
        torch.roll(torch.eye(2 * batch_size), batch_size, 1)
    neg_sample_indicators = \
        torch.ones(2 * batch_size) - torch.eye(2 * batch_size)

    if neg_pairs != "all": 
        # here, positive pairs are NOT included in the negative pairs
        min_val = 1
        max_val = torch.sum(neg_sample_indicators[0]).item() - 1
        if neg_pairs < 0:
            raise ValueError(f"Cannot use a negative amount of negative pairs "
                f"({neg_pairs}).")
        elif neg_pairs < 1:
            num_retain = int(neg_pairs * len(neg_sample_indicators))
        else:
            num_retain = int(neg_pairs)
        
        if num_retain < min_val:
            warnings.warn("Increasing the number of negative pairs to use per "
                f"image in the contrastive loss from {num_retain} to the "
                f"minimum value of {min_val}.")
            num_retain = min_val
        elif num_retain > max_val: # retain all
            num_retain = max_val

        # randomly identify the values to retain for each column
        exclusion_indicators = \
            torch.absolute(1 - neg_sample_indicators) + pos_sample_indicators
        random_values = \
            torch.rand_like(neg_sample_indicators) + \
                exclusion_indicators * 100
        retain_bool = (torch.argsort(
            torch.argsort(random_values, axis=1), axis=1
            ) < num_retain)

        neg_sample_indicators *= retain_bool
        if not (torch.sum(neg_sample_indicators, dim=1) == num_retain).all():
            raise NotImplementedError("Implementation error. Not all images "
                f"have been assigned {num_retain} random negative pair(s).")

    numerator = torch.sum(
        torch.exp(similarity_mat / temperature) * pos_sample_indicators.to(device), 
        dim=1
        )

    denominator = torch.sum(
        torch.exp(similarity_mat / temperature) * neg_sample_indicators.to(device), 
        dim=1
        )
    
    if (denominator < 1e-8).any(): # clamp, just in case
        denominator = torch.clamp(denominator, 1e-8)

    loss = torch.mean(-torch.log(numerator / denominator))
    
    return loss


def train_simclr(encoder, dataset, train_sampler, num_epochs=50, 
                 batch_size=1000, neg_pairs="all", use_cuda=True, 
                 loss_fct=None, verbose=False):
    """
    Function to train an encoder using the SimCLR loss.
    
    train_simclr(encoder, dataset, train_sampler)

    Required args:
    - encoder (nn.Module): Encoder network instance for extracting features. 
        Should have method get_features().
    - dataset (dSpritesTorchDataset): dSprites torch dataset
    - train_sampler (SubsetRandomSampler): Training dataset sampler.
    
    Optional args:
    - num_epochs (int): Number of epochs over which to train the classifier. 
        (default: 50)
    - batch_size (int): Batch size. (default: 1000)
    - neg_pairs (str or num): If "all", all available negative pairs are used 
        for the loss calculation. Otherwise, the number or proportion 
        specified by the parameter is randomly sampled and used, e.g. 5 for 5 
        examples or 0.05 for 5% of negative pairs. 
        (default: "all")
    - use_cuda (bool): If True, cuda is used, if available. (default: True)
    - loss_fct (function): loss function. If None, default contrastive loss is 
        used. (default: None)
    - verbose (bool): If True, first batch RSMs are plotted at each epoch. 
        (default: False)

    Returns: 
    - encoder (nn.Module): trained encoder
    - loss_arr (list): training loss at each epoch
    """

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    reset_encoder_device = get_model_device(encoder) # record for later
    encoder = encoder.to(device)
    projector = nn.Identity().to(device)

    if not dataset.simclr:
        raise ValueError(
            "Must pass a torch dataset for which self.simclr is True."
            )

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
        )

    # Define loss and optimizers
    train_parameters = \
        list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(train_parameters, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=500
        ) 

    # Train model on training set
    reset_encoder_training = encoder.training # record for later
    encoder.train()
    projector.train()

    if neg_pairs != "all" and loss_fct is not None:
        raise ValueError("If neg_pairs is not 'all', must use default "
            "loss function by passing None to loss_fct.")

    loss_arr = []
    for epoch_n in tqdm(range(num_epochs)):
        total_loss = 0
        num_total = 0
        for batch_idx, (X_aug1, X_aug2, Y, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            features_aug1 = encoder(X_aug1.to(device))
            features_aug2 = encoder(X_aug2.to(device))
            z_aug1 = projector(features_aug1)
            z_aug2 = projector(features_aug2)
            if loss_fct is None:
                loss = contrastive_loss(z_aug1, z_aug2, neg_pairs=neg_pairs)
            else:
                try:
                    loss = loss_fct(z_aug1, z_aug2)
                except Exception as err:
                    err.args = (
                        f"{err.args[0]} (Raised by custom loss function.)", 
                        )
                    raise err
            total_loss += loss.item()
            num_total += len(z_aug1)
            loss.backward()
            optimizer.step()
            if verbose and batch_idx == 1 and not ((epoch_n + 1) % 10):
                sorter = np.argsort(Y)
                sorted_targets = Y[sorter]
                stacked_rsm = data.calculate_torch_RSM(
                    features_aug1.detach()[sorter], features_aug2.detach()[sorter], 
                    stack=True
                    ).cpu().numpy()

                title = (f"Features (augm. 1 / augm. 2): Epoch {epoch_n} "
                    f"(batch {batch_idx})")
                sorted_target_values = \
                    dataset.dSprites.get_latent_values_from_classes(
                        sorted_targets, dataset.target_latent
                    ).squeeze()
                sorted_target_values = np.tile(sorted_target_values, 2)
                data.plot_dsprites_RSMs(
                    dataset.dSprites, stacked_rsm, sorted_target_values, 
                    titles=title, sorting_latent=dataset.target_latent
                    )
        
        loss_arr.append(total_loss / num_total)
        scheduler.step()

    projector.cpu()
    if reset_encoder_training:
        encoder.train()
    else:
        encoder.eval() 
    encoder.to(reset_encoder_device)

    return encoder, loss_arr


class VAE_decoder(nn.Module):
    def __init__(self, feat_size=84, output_dim=(1, 64, 64)):
        """
        Initializes the VAE decoder network.

        Optional args:
        - feat_size (int): size of the final features layer (default: 84)
        - output_dim (tuple): output image dimensions (channels, width, height) 
            (default: (1, 64, 64))
        """

        super().__init__()
        self.feat_size = feat_size
        self._vae = True
        self.output_dim = output_dim

        self.decoder_linear = nn.Sequential(
              nn.Linear(self.feat_size, 84),
              nn.ReLU(),
              nn.BatchNorm1d(84, affine=False),
              nn.Linear(84, 120),
              nn.ReLU(),
              nn.BatchNorm1d(120, affine=False),
              nn.Linear(120, 2704),
              nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
              nn.UpsamplingNearest2d(scale_factor=2),
              nn.BatchNorm2d(16, affine=False),
              nn.ConvTranspose2d(
                  in_channels=16, out_channels=6, kernel_size=5, stride=1
                  ),
              nn.ReLU(),
              nn.UpsamplingNearest2d(scale_factor=2),
              nn.BatchNorm2d(6, affine=False),
              nn.ConvTranspose2d(
                  in_channels=6, out_channels=1, kernel_size=5, stride=1
                  )
        )

        self._test_output_dim()

    @property
    def vae(self):
        return self._vae

    def _test_output_dim(self):
        dummy_tensor = torch.ones(1, self.feat_size)
        reset_training = self.training
        self.eval()
        with torch.no_grad():
            decoder_output_shape = self.reconstruct(dummy_tensor).shape[1:]
        if decoder_output_shape != self.output_dim:
            raise ValueError(f"Decoder produces output of shape "
                f"{decoder_output_shape} instead of expected "
                f"{self.output_dim}.")
        if reset_training:
            self.train()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.decoder_linear(z)
        h3 = h3.view(-1, 16, 13, 13)
        recon_x_logits = self.decoder_conv(h3)
        return recon_x_logits

    def forward(self, mu, logvar):
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        recon_x_logits = self.decode(z)
        return recon_x_logits, mu, logvar

    def reconstruct(self, mu):
        with torch.no_grad():
            recon_x = torch.sigmoid(self.decode(mu))
        return recon_x


def vae_loss_function(recon_X_logits, X, mu, logvar, beta=1.0):
    """
    vae_loss_function(recon_X_logits, X, mu, logvar)

    Returns the weighted VAE loss for the batch.

    Required args:
    - recon_X_logits (4D tensor): logits of the X reconstruction 
        (batch_size x shape of x)
    - X (4D tensor): X (batch_size x shape of x)
    - mu (2D tensor): mu values (batch_size x number of features)
    - logvar (2D tensor): logvar values (batch_size x number of features)

    Optional args:
    - beta (float): parameter controlling weighting of KLD loss relative to 
        reconstruction loss. (default: 1.0)
    
    Returns:
    - (float): weighted VAE loss
    """

    BCE = torch.nn.functional.binary_cross_entropy_with_logits(
        recon_X_logits, X, reduction="sum"
        )
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + beta * KLD


def train_vae(encoder, dataset, train_sampler, num_epochs=100, batch_size=500, 
              beta=1.0, use_cuda=True, verbose=False):
    """
    train_vae(encoder, dataset, train_sampler)

    Function to train an encoder using the SimCLR loss.
    
    Required args:
    - encoder (nn.Module): Encoder network instance for extracting features. 
        Should have method get_features().
    - dataset (dSpritesTorchDataset): dSprites torch dataset
    - train_sampler (SubsetRandomSampler): Training dataset sampler.
    
    Optional args:
    - num_epochs (int): Number of epochs over which to train the classifier. 
        (default: 10)
    - batch_size (int): Batch size. (default: 100)
    - beta (float): parameter controlling weighting of KLD loss relative to 
        reconstruction loss. (default: 1.0)
    - use_cuda (bool): If True, cuda is used, if available. (default: True)
    - verbose (bool): If True, 5 first batch reconstructions are plotted at 
        each epoch. (default: False)

    Returns: 
    - encoder (nn.Module): trained encoder
    - decoder (nn.Module): trained decoder
    - loss_arr (list): training loss at each epoch
    """

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    reset_encoder_device = get_model_device(encoder) # for later
    encoder = encoder.to(device)
    decoder = VAE_decoder(encoder.feat_size, encoder.input_dim).to(device)

    if not encoder.vae:
        raise ValueError("Must pass encoder for which self.vae is True.")

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
        )

    # Define loss and optimizers
    train_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(train_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=500
        )

    # Train model on training set
    reset_encoder_training = encoder.training
    encoder.train()
    decoder.train()

    loss_arr = []
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        num_total = 0
        for batch_idx, (X, _, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            recon_X_logits, mu, logvar = decoder(*encoder(X.to(device)))
            loss = vae_loss_function(
                recon_X_logits=recon_X_logits, X=X.to(device), mu=mu, 
                logvar=logvar, beta=beta
                )
            total_loss += loss.item()
            num_total += len(recon_X_logits)
            loss.backward()
            optimizer.step()
            if verbose and epoch % 10 == 9 and batch_idx == 0:
                num_images = 5
                encoder.eval()
                decoder.eval()
                with torch.no_grad():
                    input_imgs = X[:num_images].detach().cpu().numpy()
                    output_imgs = decoder.reconstruct(
                        encoder.get_features(X[:num_images].to(device))
                        ).detach().cpu().numpy()
                encoder.train()
                decoder.train()

                title = (f"Epoch {epoch}, batch {batch_idx}, "
                    f"loss {loss.item():.2f}")
                plot_util.plot_dsprite_image_doubles(
                    list(input_imgs), list(output_imgs), "Reconstr.",
                    title=title)

        loss_arr.append(total_loss / num_total)
        scheduler.step()

    # set final decoder state and reset original encoder state
    decoder.train()
    decoder.cpu()
    if reset_encoder_training:
        encoder.train()
    else:
        encoder.eval()
    encoder.to(reset_encoder_device)

    return encoder, decoder, loss_arr


def plot_vae_reconstructions(encoder, decoder, dataset, indices, title=None, 
                             use_cuda=True):
    """
    plot_vae_reconstructions(encoder, decoder, dataset, indices)

    Plots VAE reconstructions from an encoder and decoder.

    Required args:
    - encoder (CoreEncoder): encoder with self.vae set to True.
    - decoder (VAE_decoder): VAE decoder
    - dataset (dSpritesTorchDataset): dSprites torch dataset
    - indices (array-like): dataset indices to plot

    Optional args:
    - title (str): Plot title. (default: None)
    - use_cuda (bool): If True, cuda is used, if available. (default: True)
    """

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    if not (encoder.vae and decoder.vae):
        raise ValueError(
            "Must pass encoder and decoder for which self.vae is True."
            ) 

    reset_encoder_device = get_model_device(encoder) # record for later
    reset_decoder_device = get_model_device(decoder)

    # Send to device
    encoder = encoder.to(device)
    decoder = decoder.to(device)


    reset_encoder_training = encoder.train() # record for later
    reset_decoder_training = decoder.train()

    # Retrieve reconstructions in eval mode
    encoder.eval()
    decoder.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1000, sampler=indices
        )

    Xs, recon_Xs = [], []
    for X, _, _ in dataloader:
        with torch.no_grad():
            recon_X = decoder.reconstruct(
                encoder.get_features(X.to(device))
                ).detach()
            Xs.extend(list(X.cpu().numpy()))
            recon_Xs.extend(list(recon_X.cpu().numpy()))

    # reset original encoder and decoder states
    if reset_encoder_training:
        encoder.train()
    encoder.to(reset_encoder_device)
    
    if reset_decoder_training:
        decoder.train()  
    decoder.to(reset_decoder_device)
    
    plot_util.plot_dsprite_image_doubles(
        list(Xs), list(recon_Xs), "Reconstr.", title=title
        )



def plot_model_RSMs(encoders, dataset, sampler, titles=None, 
                    sorting_latent="shape", batch_size=1000, RSM_fct=None, 
                    use_cuda=True):
    """
    plot_model_RSMs(encoders, dataset, sampler)

    Plots RSMs for different models.

    Required args:
    - encoders (list): list of EncoderCore() objects
    - dataset (dSpritesTorchDataset): dSprites torch dataset
    - sampler (SubsetRandomSampler): Sampler with the indices of images for 
        which to plot the RSM.
    
    Optional args:
    - titles (list): title for each RSM. (default: None)
    - sorting_latent (str): name of latent class/feature to sort rows 
        and columns by. (default: "shape")
    - batch_size (int): Batch size. (default: 1000)
    - RSM_fct (function): torch function to calculate RSM. If None, default 
        RSM calculation function is used. (default: None)
    - use_cuda (bool): If True, cuda is used, if available. (default: True)
    
    Returns:
    - encoder_rsms (list): list of RSMs for each encoder
    """

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    if not isinstance(encoders, list):
        encoders = [encoders]
        titles = [titles]
    
    if titles is not None and len(encoders) != len(titles):
        raise ValueError("If providing titles, must provide as many as "
            f"encoders ({len(encoders)}).")

    if hasattr(dataset, "simclr") and dataset.simclr and not dataset.simclr_mode != "test":
        warnings.warn("Using a SimCLR dataset. Since the dataset returns 2 augmentations, "
            "RSMs will be calculated for the first augmentation of each image.")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler
        )

    encoder_rsms = []
    encoder_latents = []
    for encoder in encoders:
        reset_encoder_training = encoder.training
        reset_encoder_device = get_model_device(encoder)
        if not encoder.untrained:
            encoder.eval() # otherwise untrained batch norm messes things up
        encoder = encoder.to(device)
        all_features = []
        all_latents = []
        for outs in dataloader:
            Xs = outs[0]
            indices = outs[-1]

            with torch.no_grad():
                features = encoder.get_features(Xs.to(device))
            all_features.append(features)
            all_latents.append(dataset.dSprites.get_latent_values(
                indices, latent_class_names=[sorting_latent]
            )[:, 0])

        all_features = torch.cat(all_features)
        all_latents = np.concatenate(all_latents)

        if RSM_fct is None:
            rsm = data.calculate_torch_RSM(all_features).cpu().numpy()
        else:
            try:
                rsm = RSM_fct(all_features).cpu().numpy()
            except Exception as err:
                err.args = (
                    f"{err.args[0]} (Raised by custom RSM function.)", 
                    )
                raise err

        encoder_rsms.append(rsm)
        encoder_latents.append(all_latents)

        # reset original encoder state
        if reset_encoder_training:
            encoder.train()
        else:
            encoder.eval()
        encoder.to(reset_encoder_device)

    data.plot_dsprites_RSMs(
        dataset.dSprites, encoder_rsms, encoder_latents, 
        titles=titles, sorting_latent=sorting_latent
        )

    return encoder_rsms
        


def train_clfs_by_fraction_labelled(encoder, dataset, train_sampler, 
    test_sampler, labelled_fractions=None, num_epochs=10, freeze_features=True, 
    batch_size=1000, subset_seed=None, use_cuda=True, encoder_label=None, 
    plot_accuracies=True, ax=None, title=None, plot_chance=True, color="blue", 
    marker=".", verbose=False):
    """
    train_clfs_by_fraction_labelled(encoder, dataset, train_sampler, 
        test_sampler)

    Trains classifiers on an encoder, and returns training and test accuracy     
    with different fractions of labelled data. Optionally plots the results.

    Required args:
     - encoder (nn.Module): Encoder network instance for extracting features. 
        Should have method get_features().
    - dataset (dSpritesTorchDataset): dSprites torch dataset
    - train_sampler (SubsetRandomSampler): Training dataset sampler.
    - test_sampler (SubsetRandomSampler): Test dataset sampler.
    
    Optional args:
    - labelled_fractions (list): List of fractions of the total number of 
        available labelled training data to use for training. If None, the 
        DEFAULT_LABELLED_FRACTIONS global variable is used. (default: None)
    - num_epochs (int):Number of epochs over which to train the 
        classifiers, if full dataset is used (the number used is scaled 
        for each fraction). 
        (default: 10)
    - freeze_features (bool): If True, the feature encoder is frozen and only 
        the classifier is trained. If False, the encoder is also trained. 
        (default: True)
    - batch_size (int): Batch size. (default: 1000)
    - subset_seed (int): seed for selecting data subset, if applicable 
        (default: None)
    - use_cuda (bool): If True, cuda is used, if available. (default: True)
    - encoder_label (str): Label for the encoder. (default: None)
    - plot_accuracies (bool): If True, the accuracies are plotted. 
        (default: True)
    - ax (plt Axis): pyplot axis on which to plot accuracies. If None, a new 
        axis is initalized. (default: None)
    - title (str): main plot title. (default: None)
    - plot_chance (bool): if True, chance level classifier accuracy is 
        plotted. (default: False)
    - color (str): color to use when plotting the accuracies. 
        (default: "blue")
    - marker (str): marker to use when plotting the accuracies. (default: ".")
    - verbose (bool): If True, classification accuracy is printed. 
        (default: False)

    Returns: 
    - train_acc (1D np array): final training accuracy for each fraction 
        labelled
    - test_acc (1D np array): final test accuracy for each fraction labelled
    """

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    reset_encoder_device = get_model_device(encoder)
    encoder.to(device)

    if isinstance(labelled_fractions, (int, float)):
        labelled_fractions = [labelled_fractions]

    if labelled_fractions is None:
        labelled_fractions = DEFAULT_LABELLED_FRACTIONS
        labelled_fraction_str = ", ".join(
            [str(val) for val in labelled_fractions]
            )
        if verbose:
            print("Using the following default labelled fraction values: "
                f"{labelled_fraction_str}\n")

    if len(labelled_fractions) == 0:
        raise ValueError("Include at least one value in labelled_fractions.")

    if np.min(labelled_fractions) <= 0 or np.max(labelled_fractions) > 1:
        raise ValueError(
            "all labelled_fractions must be between (0, 1) (excl, incl)"
            )

    train_acc = np.full(len(labelled_fractions), np.nan)
    test_acc = np.full(len(labelled_fractions), np.nan)

    freeze_str = "" if freeze_features else "*"

    if verbose and encoder_label is not None:
        add_str = "" if freeze_features else " and encoders"
        print(f"{encoder_label[0].capitalize()}{encoder_label[1:]} "
            f"encoder: training classifiers{add_str}{freeze_str}...")

    if not freeze_features: # retain original
        orig_encoder = copy.deepcopy(encoder)

    num_epochs_use_all = [
        int(np.ceil(num_epochs / np.sqrt(labelled_fraction)))
        for labelled_fraction in labelled_fractions
        ]

    n_fractions = len(labelled_fractions)
    for i in tqdm(range(n_fractions)):
        if not freeze_features: # obtain new fresh version
            encoder = copy.deepcopy(orig_encoder)
        _,  _, train_acc[i], test_acc[i] = train_classifier(
            encoder, dataset, train_sampler, test_sampler, 
            num_epochs=num_epochs_use_all[i], 
            fraction_of_labels=labelled_fractions[i], 
            freeze_features=freeze_features, subset_seed=subset_seed, 
            batch_size=batch_size, progress_bar=False, verbose=False
            )

    if plot_accuracies:
        if ax is None:
            _, ax = plt.subplots(1)

        labelled_fractions = np.asarray(labelled_fractions)
        sorter = np.argsort(labelled_fractions)
        sorted_labelled_fractions = labelled_fractions[sorter].tolist()

        if plot_chance:
            ax.axhline(y=100 / dataset.num_classes, ls="dashed", color="gray", 
            alpha=0.7
            )
    
        if encoder_label is not None:
            training_label = f"{encoder_label}{freeze_str} (training)"
            test_label = f"{encoder_label}{freeze_str} (test)"
        else:
            training_label = "training"
            test_label = "test"

        ax.plot(
            sorted_labelled_fractions, train_acc[sorter], ls="dashed", 
            label=training_label, color=color, marker=marker, markersize=8, 
            alpha=0.4
            )
        ax.plot(
            sorted_labelled_fractions, test_acc[sorter], lw=3, 
            label=test_label, color=color, marker=marker, markersize=8, 
            alpha=0.8
            )

        ax.set_xlabel("Fraction of labelled data used (log scale)")
        ax.set_ylabel("Classification accuracy (%)")
        ax.legend()

        from matplotlib.ticker import ScalarFormatter
        ax.set_xscale("log")
        ax.set_xticks(sorted_labelled_fractions)
        if len(sorted_labelled_fractions) < 8:
            ax.set_xticklabels(sorted_labelled_fractions)
        ax.xaxis.set_major_formatter(ScalarFormatter())

        if title is not None:
            ax.set_title(title)

    encoder.to(reset_encoder_device)

    return train_acc, test_acc


def train_encoder_clfs_by_fraction_labelled(
    encoders, dataset, train_sampler, test_sampler, labelled_fractions=None, 
    num_epochs=10, freeze_features=True, batch_size=1000, subset_seed=None, 
    use_cuda=True, encoder_labels=None, plot_accuracies=True, title=None, 
    verbose=False):

    """
    train_encoder_clfs_by_fraction_labelled(encoder, train_sampler, 
        test_sampler)

    Trains classifiers on encoders, and returns training and test accuracy 
    with different fractions of labelled data. Optionally plots the results.

    Required args:
    - encoders (list): List of encoder network instances for extracting 
        features. 
    - dataset (dSpritesTorchDataset): dSprites torch dataset.
    - train_sampler (SubsetRandomSampler): Training dataset sampler.
    - test_sampler (SubsetRandomSampler): Test dataset sampler.
    
    Optional args:
    - labelled_fractions (list): List of fractions of the total number of 
        available labelled training data to use for training. If None, the 
        DEFAULT_LABELLED_FRACTIONS global variable is used. (default: None)
    - num_epochs (int or list): Number of epochs over which to train the 
        classifiers for each encoder, if full dataset is used (the number 
        used is scaled for each fraction). (default: 10)
    - freeze_features (bool or list): If True, the feature encoder is frozen 
        and only the classifier is trained. If False, the encoder is also 
        trained. A list can be provided if the value is different from encoder 
        to encoder. (default: True)
    - batch_size (int): Batch size. (default: 1000)
    - subset_seed (int): seed for selecting data subset, if applicable 
        (default: None)
    - use_cuda (bool): If True, cuda is used, if available. (default: True)
    - encoder_label (str): Label for the encoder. (default: None)
    - plot_accuracies (bool): If True, the accuracies are plotted. 
        (default: True)
    - title (str): main plot title. (default: None)
    - verbose (bool): If True, classification accuracy is printed. 
        (default: False)

    Returns: 
    - train_accs (2D np array): final training accuracies for each 
        encoder x fraction labelled
    - test_accs (2D np array): final test accuracies for each 
        encoder x fraction labelled
    if plot_accuracies:
    - ax (plt Axis): pyplot axis on which the accuracies are plotted
    """

    colors = ["blue", "brown", "green", "red", "purple", "black", "orange"] 
    markers = ["o", "^", "P", "d", "X", "p", "*"] # 7
    if len(colors) != len(markers):
        raise NotImplementedError(
            "Implementation error: there should be as many preset colors "
            f"({len(colors)}) as markers ({len(markers)})."
            )
    if len(colors) < len(encoders):
        raise NotImplementedError(
            f"Too may encoders ({len(encoders)}) for the number of "
            f"preset colors ({len(colors)})."
            )

    if isinstance(num_epochs, list):
        if len(num_epochs) != len(encoders):
            raise ValueError("If providing num_epochs as a list, must "
                "provide as many as the number of encoders.")
    else:
        num_epochs = [num_epochs] * len(encoders)


    if isinstance(labelled_fractions, (int, float)):
        labelled_fractions = [labelled_fractions]

    if labelled_fractions is None:
        labelled_fractions = DEFAULT_LABELLED_FRACTIONS
        labelled_fraction_str = ", ".join(
            [str(val) for val in labelled_fractions]
            )
        if verbose:
            print("Using the following default labelled fraction values: "
            f"{labelled_fraction_str}\n")

    if isinstance(freeze_features, list):
        if len(freeze_features) != len(encoders):
            raise ValueError("If providing freeze_features as a list, must "
                "provide as many as the number of encoders.")
    else:
        freeze_features = [freeze_features] * len(encoders)

    if isinstance(encoder_labels, list):
        if len(encoder_labels) != len(encoders):
            raise ValueError("If providing encoder_labels, must provide as "
                "many as the number of encoders.")
    else:
        encoder_labels = [None] * len(encoders)

    ax = None
    if plot_accuracies:
        _, ax = plt.subplots(1, figsize=[9, 6])
        ax.axhline(
            y=100 / dataset.num_classes, ls="dashed", color="gray", 
            alpha=0.7, lw=3
            )
        if title is not None:
            ax.set_title(title)
        

    train_accs = np.full((len(encoders), len(labelled_fractions)), np.nan)
    test_accs = np.full((len(encoders), len(labelled_fractions)), np.nan)
    for e, encoder in enumerate(encoders):
        train_accs[e], test_accs[e] = train_clfs_by_fraction_labelled(
            encoder, dataset, train_sampler, test_sampler, 
            labelled_fractions=labelled_fractions, num_epochs=num_epochs[e], 
            freeze_features=freeze_features[e], batch_size=batch_size, 
            subset_seed=subset_seed, use_cuda=use_cuda, 
            encoder_label=encoder_labels[e], plot_accuracies=plot_accuracies, 
            ax=ax, plot_chance=False, color=colors[e], marker=markers[e], 
            verbose=verbose
            )
    
    if plot_accuracies:
        return train_accs, test_accs, ax
    else:
        return train_accs, test_accs


