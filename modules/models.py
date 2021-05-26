import numpy as np
import torch
from torch import nn
import torchvision
import tqdm

from . import data


class EncoderCore(nn.Module):
    def __init__(self, feat_size=84, input_dim=(1, 64, 64)):
        """
        Initializes the core encoder network.

        Optional args:
        - feat_size (int): size of the final features layer (default: 84)
        - input_dim (tuple): input image dimensions (channels, width, height) 
            (default: (1, 64, 64))
        """

        super().__init__()

        # check input dimensions provided
        self.input_dim = tuple(input_dim)
        if len(self.input_dim) == 3:
            self.input_ch = self.input_dim[0]
        elif len(self.input_dim) == 2:
            self.input_ch = 1
        else:
            raise ValueError("input_dim should have length 2 (wid x hei) or "
              "3 (ch x wid x hei).")

        # convolutional component of the feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=self.input_ch, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(6,affine=False),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(16,affine=False)
        )

        # calculate size of the convolutional feature extractor output
        self.feat_extr_output_size = self._get_feat_extr_output_size(self.input_dim)
        self.feat_size = feat_size

        # linear component of the feature extractor
        self.linear_projections = nn.Sequential(
            nn.Linear( self.feat_extr_output_size, 120),
            nn.ReLU(),
            nn.BatchNorm1d(120, affine=False),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.BatchNorm1d(84, affine=False),
            nn.Linear(84, self.feat_size),
            nn.ReLU(),
            nn.BatchNorm1d(84, affine=False)
        )

    def _get_feat_extr_output_size(self, input_dim):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_dim)
            output = self.feature_extractor(dummy_input).shape
        return np.product(output)

    def forward(self,X):
        feats = self.feature_extractor(X)
        feats = torch.flatten(feats, 1)
        feats = self.linear_projections(feats)
        return feats

    def get_features(self,X):
        with torch.no_grad():
            feats = self.feature_extractor(X)
            feats = torch.flatten(feats, 1)
            feats = self.linear_projections(feats)
        return feats
        

def train_classifier(encoder, train_data, test_data, num_classes=3, 
                     num_epochs=10, fraction_of_labels=1.0, batch_size=1000, 
                     freeze_features=True, verbose=True, subset_seed=None, 
                     use_cuda=True):
    """
    Function to train a linear classifier to predict classes from features.
    
    Required args:
    - encoder (nn.Module): Encoder network instance for extracting features. 
        Should have method get_features().
    - train_data (torch dataset): Dataset comprising the training data.
    - test_data (torch dataset): Dataset comprising the test data.
    
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
    - use_cuda (bool): If True, cuda is used, if available. (default: True)
    - subset_seed (int): seed for selecting data subset, if applicable 
        (default: None)
    - verbose (bool): If True, classification accuracy is printed. 
        (default: True)

    Returns: 
    - classifier (nn.Linear): trained classification layer
    - loss_arr (list): training loss at each epoch
    - train_acc (float): final training accuracy
    - test_acc (float): final test accuracy
    """


    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    encoder = encoder.to(device)
    classifier = nn.Linear(encoder.feat_size, num_classes).to(device)

    simclr = True if train_data.dataset.simclr else False

    # Define datasets and dataloaders
    train_data_subset, _ = data.train_test_split_idx(
        train_data, fraction_train=fraction_of_labels, randst=subset_seed
        ) # obtain subset
    train_dataloader = torch.utils.data.DataLoader(
        train_data_subset, batch_size=batch_size, shuffle=True
        )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size
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
    loss_arr = []
    for _ in tqdm(range(num_epochs)):
        total_loss = 0
        num_total = 0
        for iter_data in train_dataloader:
            if simclr:
                X, _, y = iter_data # ignore augmented X
            else:
                X, y = iter_data
            
            classification_optimizer.zero_grad()

            if freeze_features:
                features = encoder.get_features(X.to(device))
            else:
                features = encoder(X.to(device))

            predicted_y_logits = classifier(features)
            loss = loss_fn(predicted_y_logits, y.to(device))
            loss.backward()
            classification_optimizer.step()

            total_loss += loss.item()
            num_total += y.size(0)

        loss_arr.append(total_loss / num_total)
        scheduler.step()
    
    # Calculate prediction accuracy on training and test sets
    accuracies = []
    for i, dataloader in enumerate((train_dataloader, test_dataloader)):
        num_correct = 0
        num_total = 0
        for iter_data in dataloader:
            if simclr:
                X, _, y = iter_data # ignore augmented X
            else:
                X, y = iter_data

            with torch.no_grad():
                features = encoder.get_features(X.to(device))
                predicted_y_logits = classifier(features)
            
            # identify predicted classes from logits
            _, predicted_y = torch.max(predicted_y_logits, 1)
            num_correct += (predicted_y.cpu() == y).sum()
            num_total += y.size(0)
            
        accuracy = (100 * num_correct.numpy()) / num_total
        accuracies.append(accuracy)

    train_acc, test_acc = accuracies

    if verbose:
        chance = 100 / num_classes
        if freeze_features:
            train_str = "classifier"
        else:
            train_str = "encoder and classifier"

        print(f"Network performance after {num_epochs} {train_str} training "
          f"epochs (chance: {chance:.3f}%):\n"
          f"    Training accuracy: {train_acc:.3f}%\n"
          f"    Testing accuracy: {test_acc:.3f}%")

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


def train_simclr(encoder, train_data, num_epochs=10, batch_size=1000, 
                 use_cuda=True, verbose=True, target_latent="shape"):
    """
    Function to train an encoder using the SimCLR loss.
    
    Required args:
    - encoder (nn.Module): Encoder network instance for extracting features. 
        Should have method get_features().
    - train_data (torch dataset): Dataset comprising the training data.
    
    Optional args:
    - num_epochs (int): Number of epochs over which to train the classifier. 
        (default: 10)
    - batch_size (int): Batch size. (default: 1000)
    - use_cuda (bool): If True, cuda is used, if available. (default: True)
    - verbose (bool): If True, first batch RSMs are plotted at each epoch. 
        (default: True)

    Returns: 
    - encoder (nn.Module): trained encoder
    - loss_arr (list): training loss at each epoch
    """

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    encoder = encoder.to(device)
    projector = nn.Identity().to(device)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
        )
    
    # retrieve dSprites info
    dSprites = train_data.dataset.dSprites
    
    # Define loss and optimizers
    train_parameters = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(train_parameters, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    loss_fn = contrastiveLoss

    # Train model on training set
    loss_arr = []
    for epoch_n in tqdm(range(num_epochs)):
        total_loss = 0
        for batch_idx, (X, X_aug, Y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            features = encoder(X.to(device))
            features_aug = encoder(X_aug.to(device))
            z = projector(features)
            z_aug = projector(features_aug)
            loss = loss_fn(z, z_aug)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if verbose and batch_idx == 1 and not ((epoch_n + 1) % 10):
                with torch.no_grad():
                    sorter = np.argsort(Y)
                    sorted_targets = Y[sorter]
                    stacked_rsm = data.calculate_torch_rsm(
                        features[sorter], features_aug[sorter], stack=True
                        ).cpu().numpy()

                    title = f"Features: Epoch {epoch_n} (batch {batch_idx})"
                    sorted_target_values = dSprites.get_latent_values_from_classes(
                        sorted_targets, target_latent
                        ).squeeze()
                    sorted_target_values = np.repeat(sorted_target_values, 2)
                    data.plot_dsprites_rsms(
                        dSprites, stacked_rsm, sorted_target_values, 
                        titles=title, target_latent=target_latent
                        )
        
        loss_arr.append(total_loss/len(train_dataloader))
        scheduler.step()

    return encoder, loss_arr

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

# def train_logistic_regression(model_type="vgg", target_feature="shape", 
#                               num_epochs=10, batch_size=64, verbose=True):
    
#     X_flat_size = DSPRITES_DICT["imgs"][0].size
#     if model_type in ["vgg", "vgg_untrained"]:
#         pretrained = True
#         if model_type == "vgg_untrained":
#             pretrained = False
#         model = load_vgg_classifier(target_feature=target_feature, pretrained=pretrained)
#         transform = "to_RGB"
#         classifier = model.classifier
#     elif model_type in ["resnet", "resnet_untrained"]:
#         pretrained = True
#         if model_type == "resnet_untrained":
#             pretrained = False
#         model = load_resnet_classifier(target_feature=target_feature, pretrained=pretrained)
#         transform = "to_RGB_size"
#         classifier = model.classifier
#     elif model_type == "raw_logreg":
#         model = init_logreg_classifier(target_feature=target_feature)
#         transform = None
#         classifier = model
#     elif model_type in ["vae", "ae"]:
#         model = load_v_ae_classifier(target_feature=target_feature, model_type=model_type)
#         transform = None
#         classifier = model.classifier
#     elif model_type == "simclr":
#         model = load_simclr_classifier(target_feature=target_feature)
#         transform = "simclr"
#         classifier = model.classifier
#     else:
#         raise ValueError(f"Model type {model_type} not recognized. "
#             "Must be 'vgg', 'vgg_untrained', 'vae', 'ae', 'simclr' or 'raw_logreg'.")

#     # Retrieve dataloaders     
#     train_dataloader, test_dataloader = init_dataloaders(
#         target_feature=target_feature, batch_size=batch_size, 
#         transform=transform, num_workers=0, seed=SEED
#         )

#     # Define loss and optimizers    
#     model.to(DEVICE)
#     classification_optimizer = torch.optim.Adam(
#         classifier.parameters(), lr=1e-3
#         )
#     loss_fn = torch.nn.CrossEntropyLoss()

#     model.train()

#     # Train logistic regression on training set
#     if verbose:
#         print(f"Training logistic regression over {num_epochs} epochs...")
        
#     for epoch_num in range(num_epochs):
#         for i, (X, y) in enumerate(train_dataloader):
#             classification_optimizer.zero_grad()
#             if model_type in ["raw_logreg", "vae", "ae"]:
#                 X = X.view(-1, X_flat_size)
#             y = y.to(DEVICE)
#             if model_type == "simclr":
#                 _, _, y_pred, _ = model(X.to(DEVICE), X.to(DEVICE))    
#             else:
#                 y_pred = model(X.to(DEVICE))
#             loss = loss_fn(y_pred, y)
#             loss.backward()
#             classification_optimizer.step()
  
#     # Calculate prediction accuracy on training set and test set
#     accuracies = []
#     model.eval()
#     for dataloader in [train_dataloader, test_dataloader]:
#         correct = 0
#         total = 0
#         for (X, y) in dataloader:
#             if model_type in ["raw_logreg", "vae", "ae"]:
#                 X = X.view(-1, X_flat_size)
#             y = y.to(DEVICE)
#             with torch.no_grad():
#                 if model_type == "simclr":
#                     _, _, y_pred, _ = model(X.to(DEVICE), X.to(DEVICE))    
#                 else:
#                     y_pred = model(X.to(DEVICE))
#             _, y_pred_class = torch.max(y_pred, 1)
#             correct += (y_pred_class == y).sum()
#             total += y.size(0)
#         acc = (100 * correct.to("cpu").numpy()) / total
#         accuracies.append(acc)
      
#     train_accuracy, test_accuracy = accuracies

#     if verbose:
#         chance = 100 / train_dataloader.dataset.num_outputs
#         print(f"\nResults (chance: {chance:.2f}%):\n"
#         f"    Training accuracy: {train_accuracy:.2f}%\n"
#         f"    Testing accuracy: {test_accuracy:.2f}%")

#     return model, train_accuracy, test_accuracy

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