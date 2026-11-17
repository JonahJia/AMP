from typing import Tuple, Dict, List
import torch.nn as nn
import utils.torchvision_compat  # noqa: F401
import torchvision.models as models

class ToyCNN(nn.Module):
    """A tiny CNN used for smoke tests and debugging."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)

class BackboneWrapper(nn.Module):
    """Wrapper for backbone networks to track layer information."""
    
    def __init__(self, features: nn.Module, layer_info: Dict[str, List]):
        super().__init__()
        self.features = features
        self._filter_sizes = layer_info['filter_sizes']
        self._strides = layer_info['strides']
        self._paddings = layer_info['paddings']
    
    def forward(self, x):
        return self.features(x)
    
    def get_layer_info(self) -> Dict[str, List]:
        """Get layer information for receptive field computation."""
        return {
            'filter_sizes': self._filter_sizes,
            'strides': self._strides,
            'paddings': self._paddings
        }

def _extract_vgg_features(model: nn.Module) -> Tuple[nn.Module, Dict]:
    """Extract features from VGG model and compute layer info."""
    features = model.features
    
    filter_sizes = []
    strides = []
    paddings = []
    
    for module in features:
        if isinstance(module, nn.Conv2d):
            filter_sizes.append(module.kernel_size[0])
            strides.append(module.stride[0])
            paddings.append(module.padding[0])
        elif isinstance(module, nn.MaxPool2d):
            filter_sizes.append(module.kernel_size)
            strides.append(module.stride)
            paddings.append(0)
    
    layer_info = {
        'filter_sizes': filter_sizes,
        'strides': strides,
        'paddings': paddings
    }
    
    return features, layer_info

def _extract_resnet_features(model: nn.Module) -> Tuple[nn.Module, Dict]:
    """Extract features from ResNet model and compute layer info."""
    # Remove the final FC and avgpool layers
    features = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4
    )
    
    # Manually track ResNet layer information
    # This is a simplified version - actual computation would be more complex
    # for residual blocks
    layer_info = {
        'filter_sizes': [7, 3, 3, 3, 3, 3],  # Simplified
        'strides': [2, 2, 1, 2, 2, 2],       # Simplified
        'paddings': [3, 1, 0, 1, 1, 1]       # Simplified
    }
    
    return features, layer_info

def _extract_densenet_features(model: nn.Module) -> Tuple[nn.Module, Dict]:
    """Extract features from DenseNet model and compute layer info."""
    features = model.features
    
    # Simplified layer info for DenseNet
    layer_info = {
        'filter_sizes': [7, 3, 3, 3, 3],  # Simplified
        'strides': [2, 2, 1, 2, 2],       # Simplified
        'paddings': [3, 1, 1, 1, 1]       # Simplified
    }
    
    return features, layer_info

def get_backbone(
    architecture: str,
    pretrained: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Get backbone network for feature extraction.
    
    Args:
        architecture: Name of architecture ('vgg16', 'resnet50', 'toy_cnn', etc.)
        pretrained: Whether to load pretrained weights
    
    Returns:
        Tuple of (backbone_module, layer_info_dict)
    """
    architecture = architecture.lower()
    
    if architecture == 'toy_cnn':
        model = ToyCNN()
        features = model.features
        layer_info = {
            'filter_sizes': [3, 3, 3],
            'strides': [2, 2, 2],
            'paddings': [1, 1, 1],
        }

    # VGG family
    elif architecture == 'vgg11':
        model = models.vgg11(pretrained=pretrained)
        features, layer_info = _extract_vgg_features(model)
    elif architecture == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=pretrained)
        features, layer_info = _extract_vgg_features(model)
    elif architecture == 'vgg13':
        model = models.vgg13(pretrained=pretrained)
        features, layer_info = _extract_vgg_features(model)
    elif architecture == 'vgg13_bn':
        model = models.vgg13_bn(pretrained=pretrained)
        features, layer_info = _extract_vgg_features(model)
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        features, layer_info = _extract_vgg_features(model)
    elif architecture == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=pretrained)
        features, layer_info = _extract_vgg_features(model)
    elif architecture == 'vgg19':
        model = models.vgg19(pretrained=pretrained)
        features, layer_info = _extract_vgg_features(model)
    elif architecture == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=pretrained)
        features, layer_info = _extract_vgg_features(model)
    
    # ResNet family
    elif architecture == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        features, layer_info = _extract_resnet_features(model)
    elif architecture == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        features, layer_info = _extract_resnet_features(model)
    elif architecture == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        features, layer_info = _extract_resnet_features(model)
    elif architecture == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        features, layer_info = _extract_resnet_features(model)
    elif architecture == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
        features, layer_info = _extract_resnet_features(model)
    
    # DenseNet family
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        features, layer_info = _extract_densenet_features(model)
    elif architecture == 'densenet161':
        model = models.densenet161(pretrained=pretrained)
        features, layer_info = _extract_densenet_features(model)
    elif architecture == 'densenet169':
        model = models.densenet169(pretrained=pretrained)
        features, layer_info = _extract_densenet_features(model)
    elif architecture == 'densenet201':
        model = models.densenet201(pretrained=pretrained)
        features, layer_info = _extract_densenet_features(model)
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return features, layer_info
