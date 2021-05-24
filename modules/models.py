from _typeshed import ReadableBuffer
import torch
import torchvision

def init_logreg_classifier(num_inputs, num_outputs):
    
    model = torch.nn.Sequential(torch.nn.Linear(num_inputs, num_outputs))

    return model


class ResNet18Classifier(torchvision.models.resnet.ResNet):

    def __init__(self, num_outputs=1, pretrained=True, freeze_encoder=True):
        
        self.super().__init__()

        resnet18 = torchvision.models.resnet18(pretrained=pretrained, progress=False)
        self.__dict__.update(resnet18.__dict__)
        
        self.pretrained = pretrained

        self._define_encoder()
        self.classifier = torch.nn.Sequential(
            init_logreg_classifier(self.num_encoder_outputs, num_outputs)
            )

        if freeze_encoder:
            self.freeze_encoder()

    # def _define_encoder(self):
    #     self.encoder = torch.nn.Sequential(
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