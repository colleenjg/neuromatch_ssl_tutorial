import os

import torch
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
            num_epochs = 50
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
            num_epochs = 125
    else:
        raise ValueError("Recorded model types only include 'vae' and "
            f"'simclr', but not '{model_type}'.")
        
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