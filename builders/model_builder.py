from model.DSANet import DSANet
from model.SPFNet import SPFNet
from model.SSFPN import SSFPN


def build_model(model_name,num_classes):
    if model_name == 'DSANet':
        return DSANet(classes=num_classes)
    elif model_name == 'SPFNet':
        return SPFNet("resnet18",pretrained=True,classes=num_classes)
    elif model_name == 'SSFPN':
        return SSFPN("resnet18",pretrained=True,classes=num_classes)
    else:
        raise NotImplementedError
    
