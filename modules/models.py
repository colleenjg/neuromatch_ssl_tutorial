import numpy as np
import torch
from torch import nn
import torchvision


class EncoderCore(nn.Module):
    def __init__(self, feat_size=84, input_dim=(1, 64, 64)):
        """
        Initialized the core encoder network.

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
        


class ResNet18Classifier(torchvision.models.resnet.ResNet):

    def __init__(self, num_outputs=1, pretrained=True, freeze_encoder=True):
        
        self.super().__init__()

        resnet18 = torchvision.models.resnet18(pretrained=pretrained, progress=False)
        self.__dict__.update(resnet18.__dict__)
        
        self.pretrained = pretrained

        self._define_encoder()
        self.classifier = nn.Sequential(
            init_logreg_classifier(self.num_encoder_outputs, num_outputs)
            )

        if freeze_encoder:
            self.freeze_encoder()

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

    def forward_clf(self, x):
        z = self.encoder(x)
        z_flat = torch.flatten(z, 1)
        y = self.classifier(z_flat)

        return y

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True



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

