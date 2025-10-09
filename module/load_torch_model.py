# python import
import logging
from typing import Union
# package import
import torch
import monai
# local import
from custom import models as custom_models
from utils.load_module import get_unique_attr_across

logger = logging.getLogger(__name__)

DEFAULT_MODULE_LIST = [
    monai.networks.nets, monai.losses, torch.nn,  # official models
    custom_models  # custom models
]
__all__ = ["load_model"]

def load_model(model_name: Union[str, list[str]],
               model_params: Union[dict[str, ...], list[dict[str, ...]]]):
    assert isinstance(model_name, list) == isinstance(model_params, list), \
        "model and model_params must either both be lists or neither be lists"

    if isinstance(model_name, str):
        loaded_model = get_unique_attr_across(DEFAULT_MODULE_LIST, {model_name: model_params})[0]
    else:
        assert len(model_name) == len(model_params), "model and model_params lists must be of the same length"
        loaded_model = torch.nn.ModuleList(
            [get_unique_attr_across(DEFAULT_MODULE_LIST, {name: params})[0]
             for name, params in zip(model_name, model_params)]
        )
    return loaded_model


if __name__ == "__main__":
    # example usage
    model = load_model("resnet18", dict(spatial_dims=2, pretrained=False, num_classes=10))
    print(model)
    model = load_model(['resnet50', 'ResNetFeatures'],
                       [dict(spatial_dims=2, pretrained=False, num_classes=10),
                        dict(model_name='resnet50', pretrained=False)])
    print(model)
    loss = load_model('CrossEntropyLoss', dict())
    print(loss)
