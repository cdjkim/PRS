from typing import Dict
from .component import ComponentE
from .encoder import DummyEncoder
from .encoder_resnet import ResEncoder
from .classifier import MlpClassifier, MlpClassifier4, CnnClassifier, LeNetClassifier, ResNetClassifier

# encoder
E: Dict[str, ComponentE] = {
    'resnet_encoder': ResEncoder,
    'none': DummyEncoder,

    # for single label classification.
    'mlp_classifier': MlpClassifier,
    'mlp_classifier4': MlpClassifier4,
    'cnn_classifier': CnnClassifier,
    'lenet_classifier': LeNetClassifier,
    'resnet_classifier': ResNetClassifier,
}
