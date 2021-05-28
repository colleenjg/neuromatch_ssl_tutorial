import numpy as np
import torch
from torch import nn
import torchvision
from tqdm.notebook import tqdm as tqdm

from . import data, plot_util


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

        self.vae = vae
        self.untrained = True

        # check input dimensions provided
        self.input_dim = tuple(input_dim)
        if len(self.input_dim) == 2:
            self.input_dim = (1, *input_dim)            
        elif len(self.input_dim) != 3:
            raise ValueError("input_dim should have length 2 (wid x hei) or "
                "3 (ch x wid x hei).")
        self.input_ch = self.input_dim[0]

        # convolutional component of the feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=self.input_ch, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(6, affine=False),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(16, affine=False)
        )

        # calculate size of the convolutional feature extractor output
        self.feat_extr_output_size = 2704 #self._get_feat_extr_output_size(self.input_dim)
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
        self.eval()
        with torch.no_grad():   
            output = self.feature_extractor(dummy_tensor).shape
        self.train()
        return np.product(output)


    def forward(self, X):
        if self.untrained and self.training:
            self.untrained = False
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
                     verbose=False):
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
    - num_classes (int): Number of data classes for classification. (default: 3)
    - fraction_of_labels (float): Fraction of the total number of available 
        labelled training data to use for training. (default: 1.0)
    - batch_size (int): Batch size. (default: 1000)
    - freeze_features (bool): If True, the feature encoder is frozen and only 
        the classifier is trained. If False, the encoder is also trained. 
        (default: True)
    - subset_seed (int): seed for selecting data subset, if applicable 
        (default: None)
    - use_cuda (bool): If True, cuda is used, if available. (default: True)
    - verbose (bool): If True, classification accuracy is printed. 
        (default: False)

    Returns: 
    - classifier (nn.Linear): trained classification layer
    - loss_arr (list): training loss at each epoch
    - train_acc (float): final training accuracy
    - test_acc (float): final test accuracy
    """


    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    if encoder is None:
        encoder = nn.Identity().to(device)
        encoder.get_features = encoder.forward
        encoder.untrained = True
        linear_input = dataset.dSprites.images[0].size
        if not freeze_features:
            raise ValueError("freeze_features must be True if no encoder is provided.")
    else:
        linear_input = encoder.feat_size
    
    encoder = encoder.to(device)

    classifier = nn.Linear(linear_input, dataset.num_classes).to(device)

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
    if not freeze_features:
        encoder.train()
    elif not encoder.untrained:
        encoder.eval() # otherwise untrained batch norm messes things up

    loss_arr = []
    for _ in tqdm(range(num_epochs)):
        total_loss = 0
        num_total = 0
        for iter_data in train_dataloader:
            if dataset.simclr:
                X, _, y = iter_data # ignore augmented X
            else:
                X, y = iter_data
            
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
                X, _, y = iter_data # ignore augmented X
            else:
                X, y = iter_data

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


def contrastiveLoss(proj_feat1, proj_feat2, temperature=0.5):
    """
    contrastiveLoss(proj_feat1, proj_feat2)

    Returns contrastive loss, given sets of projected features, with positive 
    pairs matched along the batch dimension.

    Required args:
    - proj_feat1 (2D torch Tensor): first set of projected features 
        (batch_size x feat_size)
    - proj_feat2 (2D torch Tensor): second set of projected features 
        (batch_size x feat_size)
      
    Optional args:
    - temperature (float): relaxation temperature. (default: 0.5)

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
    positive_sample_indicators = torch.roll(torch.eye(2 * batch_size), batch_size, 1)
    negative_sample_indicators = torch.ones(2 * batch_size) - torch.eye(2 * batch_size)

    numerator = torch.sum(
        torch.exp(similarity_mat / temperature) * positive_sample_indicators.to(device), 
        dim=1
        )
    denominator = torch.sum(
        torch.exp(similarity_mat / temperature) * negative_sample_indicators.to(device), 
        dim=1
        )
    loss = torch.mean(-torch.log(numerator / denominator))
    
    return loss


def train_simclr(encoder, dataset, train_sampler, num_epochs=10, batch_size=1000, 
                 use_cuda=True, verbose=False):
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
        (default: 10)
    - batch_size (int): Batch size. (default: 1000)
    - use_cuda (bool): If True, cuda is used, if available. (default: True)
    - verbose (bool): If True, first batch RSMs are plotted at each epoch. 
        (default: False)

    Returns: 
    - encoder (nn.Module): trained encoder
    - loss_arr (list): training loss at each epoch
    """

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    encoder = encoder.to(device)
    projector = nn.Identity().to(device)

    if not dataset.simclr:
        raise ValueError("Must pass a simclr torch dataset (i.e., where "
            "dataset.simclr is True.")

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
        )

    # Define loss and optimizers
    train_parameters = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(train_parameters, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    loss_fn = contrastiveLoss

    # Train model on training set
    encoder.train()
    projector.train()

    loss_arr = []
    for epoch_n in tqdm(range(num_epochs)):
        total_loss = 0
        num_total = 0
        for batch_idx, (X, X_aug, Y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            features = encoder(X.to(device))
            features_aug = encoder(X_aug.to(device))
            z = projector(features)
            z_aug = projector(features_aug)
            loss = loss_fn(z, z_aug)
            total_loss += loss.item()
            num_total += len(z)
            loss.backward()
            optimizer.step()
            if verbose and batch_idx == 1 and not ((epoch_n + 1) % 10):
                sorter = np.argsort(Y)
                sorted_targets = Y[sorter]
                stacked_rsm = data.calculate_torch_RSM(
                    features.detach()[sorter], features_aug.detach()[sorter], 
                    stack=True
                    ).cpu().numpy()

                title = f"Features (true/augm.): Epoch {epoch_n} (batch {batch_idx})"
                sorted_target_values = dataset.dSprites.get_latent_values_from_classes(
                    sorted_targets, dataset.target_latent
                    ).squeeze()
                sorted_target_values = np.tile(sorted_target_values, 2)
                data.plot_dsprites_RSMs(
                    dataset.dSprites, stacked_rsm, sorted_target_values, 
                    titles=title, sorting_latent=dataset.target_latent
                    )
        
        loss_arr.append(total_loss / num_total)
        scheduler.step()

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
        self.vae = True
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
              nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=1),
              nn.ReLU(),
              nn.UpsamplingNearest2d(scale_factor=2),
              nn.BatchNorm2d(6, affine=False),
              nn.ConvTranspose2d(in_channels=6, out_channels=1, kernel_size=5, stride=1)
        )

        self._test_output_dim()


    def _test_output_dim(self):
        dummy_tensor = torch.ones(1, self.feat_size)
        self.eval()
        with torch.no_grad():
            decoder_output_shape = self.reconstruct(dummy_tensor).shape[1:]
        if decoder_output_shape != self.output_dim:
            raise ValueError(f"Decoder produces output of shape {decoder_output_shape} "
                f"instead of expected {self.output_dim}.")
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
    - recon_X_logits (4D tensor): logits of the X reconstruction (batch_size x shape of x)
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


def train_vae(encoder, dataset, train_sampler, num_epochs=10, batch_size=1000, 
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
    - batch_size (int): Batch size. (default: 1000)
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

    encoder = encoder.to(device)
    decoder = VAE_decoder(encoder.feat_size, encoder.input_dim).to(device)

    if not encoder.vae:
        raise ValueError("Must pass a VAE Encoder (i.e., where encoder.vae "
            "is True).")

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
        )

    # Define loss and optimizers
    train_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(train_params, lr=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    # Train model on training set
    encoder.train()
    decoder.train()

    loss_arr = []
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        num_total = 0
        for batch_idx, (X, Y) in enumerate(train_dataloader):
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
            if verbose and epoch % 10 == 9 and batch_idx == 1:
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

                title = f"Epoch {epoch}, batch {batch_idx}, loss {loss.item():.2f}"
                plot_util.plot_dsprite_image_doubles(
                    list(input_imgs), list(output_imgs), "Reconstr.",
                    title=title)

        loss_arr.append(total_loss / num_total)
        scheduler.step()

    return encoder, decoder, loss_arr


def plot_vae_reconstructions(encoder, decoder, dataset, indices, title=None, use_cuda=True):
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
        raise ValueError("encoder and decoder must have self.vae set to True.") 

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Retrieve reconstructions in eval mode
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        Xs = dataset[indices][0].to(device)
        recon_Xs = decoder.reconstruct(encoder.get_features(Xs)).detach().cpu().numpy()
        Xs = Xs.cpu().numpy()

    encoder.train()
    decoder.train()  

    plot_util.plot_dsprite_image_doubles(list(Xs), list(recon_Xs), "Reconstr.", title=title)



def plot_model_RSMs(encoders, dataset, sampler, titles=None, sorting_latent="shape", 
                    untrained=False, use_cuda=True):
    """
    plot_model_RSMs(encoders, dataset, sampler)

    Plots RSMs for different models.

    Required args:
    - encoders (list): list of EncoderCore() objects
    - dataset (dSpritesTorchDataset): dSprites torch dataset
    - sampler (SubsetRandomSampler): Sampler with the indices of images for which to 
        plot the RSM.
    
    Optional args:
    - titles (list): title for each RSM. (default: None)
    - sorting_latent (str): name of latent class/feature to sort rows 
        and columns by. (default: "shape")
    - use_cuda (bool): If True, cuda is used, if available. (default: True)
    """

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    if not isinstance(encoders, list):
        encoders = [encoders]
        titles = [titles]
    
    if titles is not None and len(encoders) != len(titles):
        raise ValueError("If providing titles, must provide as many as encoders.")

    batch_size = 1000
    n_batches = int(np.ceil(len(sampler.indices) / batch_size))

    encoder_rsms = []
    encoder_latents = []
    for encoder in encoders:
        if not encoder.untrained:
            encoder.eval() # otherwise untrained batch norm messes things up
        encoder = encoder.to(device)
        all_features = []
        all_latents = []
        for b_idx in range(n_batches):
            indices = sampler.indices[b_idx * batch_size : (b_idx + 1) * batch_size]
            if dataset.simclr:
                Xs, _, _ = dataset[indices]
            else:
                Xs, _ = dataset[indices]
            with torch.no_grad():
                features = encoder.get_features(Xs.to(device))
            all_features.append(features)
            all_latents.append(dataset.dSprites.get_latent_values(
                indices, latent_class_names=[sorting_latent]
            )[:, 0])

        all_features = torch.cat(all_features)
        all_latents = np.concatenate(all_latents)
        rsm = data.calculate_torch_RSM(all_features).cpu().numpy()

        encoder_rsms.append(rsm)
        encoder_latents.append(all_latents)

    data.plot_dsprites_RSMs(
        dataset.dSprites, encoder_rsms, encoder_latents, 
        titles=titles, sorting_latent=sorting_latent
        )
        


# class ResNet18Classifier(torchvision.models.resnet.ResNet):

#     def __init__(self, num_outputs=1, pretrained=True, freeze_encoder=True):
        
#         self.super().__init__()

#         resnet18 = torchvision.models.resnet18(pretrained=pretrained, progress=False)
#         self.__dict__.update(resnet18.__dict__)
        
#         self.pretrained = pretrained

#         self._define_encoder()
#         self.classifier = nn.Sequential(
#             init_logreg_classifier(self.num_encoder_outputs, num_outputs)
#             )

#         if freeze_encoder:
#             self.freeze_encoder()

    # def _define_encoder(self):
    #     self.encoder = nn.Sequential(
    #         self.conv1, 
    #         self.bn1,
    #         self.relu,
    #         self.maxpool,

    #         self.layer1, 
    #         self.layer2, 
    #         self.layer3, 
    #         self.layer4, 

    #         self.avgpool
        
    #             x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)

    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
        
    #     )

    # def forward_clf(self, x):
    #     z = self.encoder(x)
    #     z_flat = torch.flatten(z, 1)
    #     y = self.classifier(z_flat)

    #     return y

    # def freeze_encoder(self):
    #     for param in self.encoder.parameters():
    #         param.requires_grad = False

    # def unfreeze_encoder(self):
    #     for param in self.encoder.parameters():
    #         param.requires_grad = True



#     def feature_encoder(self):


#     # set gradients to 0
#     for param in resnet.parameters():
#         param.requires_grad = False
    
#     num_features = resnet.fc.in_features
#     num_outputs = DSPRITES_DICT["latent_num_possible_values"][target_feature]
#     features = [torch.nn.Linear(num_features, num_outputs)]
#     resnet.fc = torch.nn.Sequential(*features)
#     resnet.classifier = resnet.fc

#     return resnet


# def load_vgg_classifier(target_feature="shape", pretrained=True):
#     if pretrained:
#         vgg16 = copy.deepcopy(VGG_PRETRAINED)
#     else:
#         vgg16 = copy.deepcopy(VGG_UNTRAINED)
#     # set gradients to 0
#     for param in vgg16.features.parameters():
#         param.requires_grad = False
    
#     num_features = list(vgg16.classifier.children())[0].in_features
#     num_outputs = DSPRITES_DICT["latent_num_possible_values"][target_feature]
#     features = [torch.nn.Linear(num_features, num_outputs)]
#     vgg16.classifier = torch.nn.Sequential(*features)

#     return vgg16

# def load_v_ae_classifier(target_feature="shape", model_type="vae"):
#     if model_type == "vae":
#         model = copy.deepcopy(VAE)
#     elif model_type == "ae":
#         model = copy.deepcopy(AE)
#     else:
#         raise ValueError("model_type must be either 'vae' or 'ae'.")
#     # set gradients to 0
#     for param in model.features.parameters():
#         param.requires_grad = False
    
#     num_features = list(model.features.children())[-1].out_features
#     num_outputs = DSPRITES_DICT["latent_num_possible_values"][target_feature]
#     features = [torch.nn.Linear(num_features, num_outputs)]
#     model.classifier = torch.nn.Sequential(*features)
#     model.forward = model.forward_clf

#     return model

# def load_simclr_classifier(target_feature="shape"):

#     simclr = copy.deepcopy(SIMCLR)
#     # set gradients to 0
#     for param in simclr.encoder.parameters():
#         param.requires_grad = False
    
#     simclr.projector # replace with classifier

#     simclr.features = simclr.encoder # create aliases

#     num_features = list(simclr.projector.children())[0].in_features
#     num_outputs = DSPRITES_DICT["latent_num_possible_values"][target_feature]
#     features = [torch.nn.Linear(num_features, num_outputs)]
#     simclr.classifier = torch.nn.Sequential(*features)
#     simclr.projector = simclr.classifier

#     return simclr

# def load_pretrained_SimCLR():
#     import os
#     if not os.path.exists("SimCLR"):
#         !git clone https://github.com/spijkervet/SimCLR.git --quiet
#         !wget -o SimCLR/simclr_model.tar https://github.com/Spijkervet/SimCLR/releases/download/1.1/checkpoint_100.tar --quiet
#         !python3 -m pip install SimCLR --quiet

#     from simclr import SimCLR
#     from simclr.modules import get_resnet
#     from simclr.modules.transformations import TransformsSimCLR

#     encoder = get_resnet("resnet18", pretrained=False)
#     n_features = encoder.fc.in_features
#     SIMCLR = SimCLR(encoder, 64, n_features)
#     _ = SIMCLR.load_state_dict(torch.load("checkpoint_100.tar", map_location=torch.device(DEVICE)))

#     return SIMCLR