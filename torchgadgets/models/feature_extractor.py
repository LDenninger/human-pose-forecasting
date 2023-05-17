import torch.nn as nn
import torchvision.models as tv_models



RESNET_FEATURE_DIM = {
    18:{
        0: 1000,
        1: 512,
        2: 25088,
        3: 50176
    },
    50:{
        0: 1000,
        1: 2048,
        2: 100352,
        3: 200704
    },
    101:{
        0: 1000,
        1: 2048
    }
}

CONVNEXT_FEATURE_DIM = {
    "tiny" : {
        0: 1000,
        1: 768,
        2: 37632,
        3: 75264,
    },
    "small": {
        0: 1000,
        1: 768,
        2: 37632,

    },
    "base": {
        0: 1000,
        1: 1024,
        2: 50176
    },
    "large": {
        0: 1000,
        1: 1536,
        2: 75264
    },

}


VGG_FEATURE_DIM = {
    11: {
        0: 1000,
        1: 25088,
        2: 25088,
        3: 115200
    }
}

MOBILENETV3_FEATURE_DIM = {
    'small': {
        1: 576
    }

}

class ResNet(nn.Module):
    """
        Class that bundles the ResNet architecture in different configurations.

        Parameters:
            depth (int): Size of the ResNet architecture.
            layer (int): Number of layers to remove from the back. 
                            Default is 1, meaning only the last fully connected layer is removed.
            weights (str): Which pretrained model weights to use.
                                Default is using pretrained weights from the ImageNet-1k dataset.
    
    """

    def __init__(self, size=50, layer=1, weights='DEFAULT'):
        super(ResNet, self).__init__()
        assert size in [18, 34, 50, 101, 152], 'Please provide a valid size from: 18, 34, 50, 101, 152'
        resnet_dict = {
            18: tv_models.resnet18,
            34: tv_models.resnet34,
            50: tv_models.resnet50,
            101: tv_models.resnet101,
            152: tv_models.resnet152
        }
        resnet = resnet_dict[size](weights=weights)
        # remove last FC layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-(layer)])

    def forward(self, input_):
        out = self.resnet(input_)
        out = out.view(out.shape[0], -1)
        return out
    

    
class ConvNeXt(nn.Module):
    """
        Class that bundles the ConvNeXt architecture in different configurations.

        Parameters:
            model (str): Size of the ConvNeXt architecture.
            layer (int): Number of layers to remove from the back. 
                            Default is 1, meaning only the last fully connected layer is removed.
            weights (str): Which pretrained model weights to use.
                                Default is using pretrained weights from the ImageNet-1k dataset.
    
    """

    def __init__(self, size='tiny', layer=1, weights='DEFAULT'):
        super(ConvNeXt, self).__init__()
        assert size in ['tiny','small', 'base', 'large'], 'Please provide a valid size from: tiny, small, base, large'
        convnext_dict = {
            'tiny': tv_models.convnext_tiny,
            'small': tv_models.convnext_small,
            'base': tv_models.convnext_base,
            'large': tv_models.convnext_large
        }
        convnext = convnext_dict[size](weights=weights)

        # remove last FC layer
        self.convnext = nn.Sequential(*list(convnext.children())[:-(layer)])

    def forward(self, input_):
        out = self.convnext(input_)
        out = out.view(out.shape[0], -1)
        return out
    
class VGG(nn.Module):
    """
        Class that bundles the ResNet architecture in different configurations.

        Parameters:
            depth (int): Size of the ResNet architecture.
            layer (int): Number of layers to remove from the back. 
                            Default is 1, meaning only the last fully connected layer is removed.
            weights (str): Which pretrained model weights to use.
                                Default is using pretrained weights from the ImageNet-1k dataset.
    
    """

    def __init__(self, size=11, batch_norm=False, layer=1, weights='DEFAULT'):
        super(VGG, self).__init__()
        assert size in [11, 13, 16, 19], 'Please provide a valid size from: 11, 13, 16, 19'
        if not batch_norm:
            vgg_dict = {
                11: tv_models.vgg11,
                13: tv_models.vgg13,
                16: tv_models.vgg16,
                19: tv_models.vgg19,
            }
        else:
            vgg_dict = {
                11: tv_models.vgg11_bn,
                13: tv_models.vgg13_bn,
                16: tv_models.vgg16_bn,
                19: tv_models.vgg19_bn,
            }
        vgg = vgg_dict[size](weights=weights)
        # remove last FC layer
        self.vgg = nn.Sequential(*list(vgg.children())[:-(layer)])

    def forward(self, input_):
        out = self.vgg(input_)
        out = out.view(out.shape[0], -1)
        return out

class MobileNetV3(nn.Module):
    """
        Class that bundles the ResNet architecture in different configurations.

        Parameters:
            depth (int): Size of the ResNet architecture.
            layer (int): Number of layers to remove from the back. 
                            Default is 1, meaning only the last fully connected layer is removed.
            weights (str): Which pretrained model weights to use.
                                Default is using pretrained weights from the ImageNet-1k dataset.
    
    """

    def __init__(self, size='small', layer=1, weights='DEFAULT'):
        super(MobileNetV3, self).__init__()
        assert size in ['small', 'large'], 'Please provide a valid size from: small, large'
        mobilenet_dict = {
            'small': tv_models.mobilenet_v3_small,
            'large': tv_models.mobilenet_v3_large,
        }
 
        mobilenet = mobilenet_dict[size](weights=weights)
        # remove last FC layer
        if layer!=0:
            self.mobilenet = nn.Sequential(*list(mobilenet.children())[:-(layer)])
        else:
            self.mobilenet = nn.Sequential(*list(mobilenet.children()))

    def forward(self, input_):
        out = self.mobilenet(input_)
        out = out.view(out.shape[0], -1)
        return out
    
class VisualTransformer(nn.Module):
    def __init__(self, size='b_16', layer=1, weights='DEFAULT'):
        super(VisualTransformer, self).__init__()
        assert size in ['b_16', 'b_32', 'l_16', 'l_32', 'l_16', 'l_32', 'h_14'], 'Please provide a valid size from: small, large'
        vit_dict = {
            'b_16': tv_models.vit_b_16,
            'b_32': tv_models.vit_b_32,
            'l_16': tv_models.vit_l_16,
            'l_32': tv_models.vit_l_32,
            'l_16': tv_models.vit_l_16,
            'l_32': tv_models.vit_l_32,
            'h_14': tv_models.vit_h_14

        }
 
        vit = vit_dict[size](weights=weights)
        # remove last FC layer
        if layer!=0:
            self.vit = nn.Sequential(*list(vit.children())[:-(layer)])
        else:
            self.vit = nn.Sequential(*list(vit.children()))

    def forward(self, input_):
        out = self.vit(input_)
        out = out.view(out.shape[0], -1)
        return out