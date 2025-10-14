# python import
import logging
from typing import Union
# package import
import torch
# local import

logger = logging.getLogger(__name__)

__all__ = ["load_optimizer"]

def read_optimizer_params(model, optimizer_params: dict):
    params_list = optimizer_params.pop('params')
    lr = optimizer_params.pop('lr')
    lr_list = lr if isinstance(lr, list) else [lr] * len(params_list)
    assert len(params_list) == len(lr_list), "params and lr lists must be of the same length"
    model_id = optimizer_params.pop('model_id', None)
    assert (model_id is None) == (not isinstance(model, torch.nn.ModuleList)), \
        "model_id should be provided when model is a list of models"
    model_id_list = model_id if isinstance(model_id, list) else [model_id] * len(params_list)

    params = [getattr(model if model_id is None else model[model_id], name).parameters()
              if hasattr(getattr(model if model_id is None else model[model_id], name), 'parameters')
              else getattr(model if model_id is None else model[model_id], name)
              for lr, name, model_id in zip(lr_list, params_list, model_id_list)]

    final_optim_params = [{'params': p, 'lr': lr} for p, lr in zip(params, lr_list)]
    return final_optim_params


def load_optimizer(optimizer_name: Union[str, list[str]],
                   optimizer_params: Union[dict[str, ...], list[dict[str, ...]]],
                   model) -> Union[torch.optim.Optimizer, list[torch.optim.Optimizer]]:
    assert isinstance(optimizer_name, list) == isinstance(optimizer_params, list), \
        "optimizer and optimizer_params must either both be lists or neither be lists"

    if isinstance(optimizer_name, str) and isinstance(optimizer_params, dict):  # one optim for all models
        if optimizer_params.get('params', None) is None:  # all parameters use same lr
            loaded_optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_params)
        else:  # different parameters use different lr
            final_optim_params = read_optimizer_params(model, optimizer_params)
            loaded_optimizer = getattr(torch.optim, optimizer_name)(final_optim_params)
    else:  # multiple optim for all models
        assert len(optimizer_name) == len(optimizer_params), \
            "optimizer and optimizer_params lists must be of the same length"
        if all([opt_params.get('params', None) is None for opt_params in optimizer_params]):
            assert len(optimizer_name) == len(model), \
                "when no params provided, number of optimizers must match number of models"
            loaded_optimizer = [getattr(torch.optim, name)(m.parameters(), **opt_params)
                                for name, opt_params, m in zip(optimizer_name, optimizer_params, model)]
        else:
            assert all([opt_params.get('model_id', None) is not None for opt_params in optimizer_params]), \
                "when params provided, model_id should be provided"
            final_optim_params = [read_optimizer_params(model, opt_params) for opt_params in optimizer_params]
            loaded_optimizer = [getattr(torch.optim, name)(params) for name, params in
                                zip(optimizer_name, final_optim_params)]
    return loaded_optimizer


if __name__ == "__main__":
    # one optimizer, one model
    optimizer = load_optimizer('Adam', dict(lr=0.001), torch.nn.Linear(10, 10))
    print(optimizer)

    # one optimizer, multiple models
    optimizer = load_optimizer('SGD',
                               dict(params=['weight', 'weight'], lr=0.01, model_id=[0, 1]),
                               torch.nn.ModuleList([torch.nn.Linear(10, 10),
                                                    torch.nn.Linear(10, 5)]))
    print(optimizer)

    # multiple optimizers, multiple models
    # complex case: different parameters use different lr, different models
    optimizer = load_optimizer(['Adam', 'SGD', 'AdamW'],
                               [dict(params=['weight'], lr=0.001, model_id=0),
                                dict(params=['weight'], lr=0.01, model_id=1),
                                dict(params=['bias', 'bias'], lr=[0.002, 0.02], model_id=[0, 1])],
                               torch.nn.ModuleList([torch.nn.Linear(10, 10),
                                                    torch.nn.Linear(10, 5)]))
    print(optimizer)
    # simple case: all parameters use same lr, different models
    optimizer = load_optimizer(['Adam', 'SGD'],
                               [dict(lr=0.001), dict(lr=0.01)],
                               torch.nn.ModuleList([torch.nn.Linear(10, 10),
                                                    torch.nn.Linear(10, 5)]))
    print(optimizer)