import os

import numpy as np
import torch
import torch.nn as nn
import torchvision

from . import models


def load_encoder(save_direc, model_type="simclr", dataset_type="full", 
                 neg_pairs="all", verbose=True):
    """
    load_encoder(save_direc)

    Loads encoder (VAE or SimCLR) with pre-trained parameters.

    Required args:
    - save_direc (str): directory where pre-trained encoder parameters are 
        saved

    Optional args:
    - model_type (str): type of pre-trained encoder model (default: "simclr")
    - dataset_type (str): type of dataset used in the pre-training 
        (default: "full") 
    - neg_pairs (str or num): Number of negeative pairs used in loss 
        calculation, if loading a SimCLR model. (default: "all")
    - verbose (bool): If True, details of the encoder being loaded are 
        printed. (default: True)

    Returns:
    - encoder (models.EncoderCore): encoder loaded with pre-trained parameters
    """


    if dataset_type == "full":
        dataset_type_str = ""
    elif dataset_type in ["biased", "bias_ctrl"]:
        dataset_type_str = f"_{dataset_type}"
    else:
        raise ValueError("dataset_type can only be 'full', 'biased' or "
          f"'bias_ctrl', but found '{dataset_type}'.")

    vae = False
    simclr_transforms_str, simclr_transforms_str_pr = "", ""
    neg_str, neg_str_pr = "", ""
    
    seed = 2021
    if model_type == "vae":
        batch_size = 500
        vae = True
        model_name = "VAE"
        if dataset_type == "full":
            num_epochs = 300
        elif dataset_type in ["biased", "bias_ctrl"]:
            num_epochs = 450
        
    elif model_type == "simclr":
        batch_size = 1000
        model_name = "SimCLR"
        simclr_transforms_str_pr = ("\nwith the following random affine "
            "transforms:\n\tdegrees=90\n\ttranslation=(0.2, 0.2)"
            "\n\tscale=(0.8, 1.2)")
        simclr_transforms_str = "_deg90_trans0-2_scale0-8to1-2"
        if dataset_type == "full":
            num_epochs = 60
            if neg_pairs != "all":
                neg_pairs = int(neg_pairs)
                if neg_pairs != 2:
                    raise ValueError("If not 'all'', neg_pairs must be set "
                        "to 2, as that is the only value that was used in "
                        "the saved models.")
                neg_str_pr = (f"\nwith {neg_pairs} negative pairs per image "
                    "used in the contrastive loss, and")
                neg_str = f"_{neg_pairs}neg"
        elif dataset_type in ["biased", "bias_ctrl"]:
            if neg_pairs != "all":
                raise ValueError(
                    "No saved model for SimCLR with few negative pairs using "
                    "the biased or bias_ctrl datasets."
                    )
            num_epochs = 150
    
    elif model_type == "supervised":
        model_name = model_type
        batch_size = 1000
        num_epochs = 10

    elif model_type == "random":
        model_name = model_type
        batch_size = 0
        num_epochs = 0

    else:
        raise ValueError("Recorded model types only include 'supervised', "
            f"'random', 'vae', 'simclr', but not '{model_type}'.")
        
    encoder_path = (
        f"{model_type}_encoder{dataset_type_str}{neg_str}_{num_epochs}ep_"
        f"bs{batch_size}{simclr_transforms_str}_seed{seed}.pth"
    )
    full_path = os.path.join(save_direc, "checkpoints", encoder_path)
    
    if verbose:
        model_details = (f"    => trained for {num_epochs} epochs "
            f"(batch_size of {batch_size}) on the {dataset_type} dSprites "
            f"subset dataset{neg_str_pr}{simclr_transforms_str_pr}.")
        print(f"Loading {model_name} encoder from '{full_path}'.\n"
            f"{model_details}")

    encoder = models.EncoderCore(vae=vae)
    encoder.load_state_dict(torch.load(full_path))

    return encoder


def load_vae_decoder(save_direc, verbose=True):
    """
    load_vae_decoder(save_direc)

    Loads VAE decoder with pre-trained parameters.

    Required args:
    - save_direc (str): directory where pre-trained decoder parameters are 
        saved

    Optional args:
    - verbose (bool): If True, details of the decoder being loaded are 
        printed. (default: True)

    Returns:
    - encoder (models.VAE_decoder): decoder loaded with pre-trained parameters
    """
    
    batch_size = 500
    seed = 2021
    model_name = "VAE"
    num_epochs = 300
    decoder_path =f"vae_decoder_{num_epochs}ep_bs{batch_size}_seed{seed}.pth"

    full_path = os.path.join(save_direc, "checkpoints", decoder_path)
    
    if verbose:
        model_details = (f"   => trained for {num_epochs} epochs "
            f"(batch_size of {batch_size}) on the full dSprites subset "
            "dataset.")
        print(f"Loading {model_name} decoder from '{full_path}'.\n"
            f"{model_details}")

    decoder = models.VAE_decoder()
    decoder.load_state_dict(torch.load(full_path))

    return decoder


class ResNet18_with_encoder(torchvision.models.resnet.ResNet):
    """
    ResNet18_with_encoder()

    torchvision ResNet18 with explicitly defined encoder attribute, and 
    get_features() method.

    Optional args:
    - pretrained (bool): If True, the model is pretrained. (default: True)
    """
    
    def __init__(self, pretrained=True):

        self._untrained = not(pretrained)

        resnet18 = torchvision.models.resnet18(
            pretrained=pretrained, progress=False
            )
        self.__dict__.update(resnet18.__dict__)
        
        self.pretrained = pretrained
        self.input_dim = (3, 224, 224)

        self._define_encoder()
        self.feat_size = self._get_feat_extr_output_size(self.input_dim)

    @property
    def untrained(self):
        return self._untrained

    @property
    def vae(self):
        return False

    def _define_encoder(self):
        # first 8
        self.encoder = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )

    def _get_feat_extr_output_size(self, input_dim):
        dummy_tensor = torch.ones(1, *input_dim)
        reset_training = self.training
        self.eval()
        with torch.no_grad():   
            output_dim = self.encoder(dummy_tensor).shape
        if reset_training:
            self.train()
        return np.product(output_dim)

    def get_features(self, X):
        with torch.no_grad():
            feats = self.encoder(X)
        return feats

    def forward(self, *args, **kwargs):
        self._untrained = False
        super().forward(*args, **kwargs)


class VGG16_with_encoder(torchvision.models.vgg.VGG):
    """
    VGG16_with_encoder()

    torchvision VGG16 with explicitly defined encoder attribute, and 
    get_features() method.

    Optional args:
    - pretrained (bool): If True, the model is pretrained. (default: True)
    """

    def __init__(self, pretrained=True):

        self._untrained = not(pretrained)

        vgg16 = torchvision.models.vgg16(
            pretrained=pretrained, progress=False
            )
        self.__dict__.update(vgg16.__dict__)
        
        self.pretrained = pretrained

        self._define_encoder()
        self.input_dim = (3, 64, 64)
        self.feat_size = self._get_feat_extr_output_size(self.input_dim)

    @property
    def untrained(self):
        return self._untrained

    @property
    def vae(self):
        return False

    def _define_encoder(self):
        self.encoder = self.features # alias

    def _get_feat_extr_output_size(self, input_dim):
        dummy_tensor = torch.ones(1, *input_dim)
        reset_training = self.training
        self.eval()
        with torch.no_grad():   
            output_dim = self.encoder(dummy_tensor).shape
        if reset_training:
            self.train()
        return np.product(output_dim)

    def get_features(self, X):
        with torch.no_grad():
            feats = self.encoder(X)
        return feats

    def forward(self, *args, **kwargs):
        self._untrained = False
        super().forward(*args, **kwargs)


class SimCLR_spijk_with_encoder(nn.Module):
    """
    SimCLR_spijk_with_encoder()

    SimCLR implementation from https://github.com/Spijkervet/SimCLR, with 
    explicitly defined get_features() method.

    Optional args:
    - pretrained (bool): If True, the model is pretrained. (default: True)
    """

    def __init__(self, pretrained=True):
        
        self.projection_dim = 64
        self._untrained = not(pretrained)

        import simclr

        encoder = simclr.modules.get_resnet("resnet18", pretrained=pretrained)
        simclr_model = simclr.SimCLR(
            encoder, self.projection_dim, encoder.fc.in_features
            )
        self.__dict__.update(simclr_model.__dict__)

        self.pretrained = pretrained

        if self.pretrained:
            src = ("https://github.com/Spijkervet/SimCLR/releases/download/"
                "1.1/checkpoint_100.tar")

            state_dict = torch.hub.load_state_dict_from_url(
                src, progress=False, map_location="cpu"
                )

            self.load_state_dict(state_dict)

        self.input_dim = (3, 224, 224)
        self.feat_size = self._get_feat_extr_output_size(self.input_dim)


    @property
    def untrained(self):
        return self._untrained

    @property
    def vae(self):
        return False

    def _get_feat_extr_output_size(self, input_dim):
        dummy_tensor = torch.ones(1, *input_dim)
        reset_training = self.training
        self.eval()
        with torch.no_grad():   
            output_dim = self.encoder(dummy_tensor).shape
        if reset_training:
            self.train()
        return np.product(output_dim)

    def get_features(self, X):
        with torch.no_grad():
            feats = self.encoder(X)
        return feats

    def forward(self, *args, **kwargs):
        self._untrained = False
        super().forward(*args, **kwargs)
