# python import
import logging
from typing import Union
# package import
import torch
# local import
from custom import lr_schedulers as custom_lr_scheduler
from utils.load_module import get_unique_attr_across

logger = logging.getLogger(__name__)


def load_lr_scheduler(lr_scheduler_name: Union[str, list[str], None],
                      lr_scheduler_params: Union[dict[str, ...], list[dict[str, ...]], None],
                      lr_scheduler_other_params: Union[dict[str, ...], list[dict[str, ...]], None],
                      optimizer: Union[torch.optim.Optimizer, list[torch.optim.Optimizer], None]):
    scheduler_lt = [lr_scheduler_name, lr_scheduler_params, lr_scheduler_other_params]
    assert all(var is None for var in scheduler_lt) or all(var is not None for var in scheduler_lt), \
    'if lr_scheduler is valid, lr_scheduler_params and lr_scheduler_other_params must be provided'

    if lr_scheduler_name is None:
        loaded_lr_scheduler = None
    else:
        lr_scheduler_func = get_unique_attr_across([torch.optim.lr_scheduler, custom_lr_scheduler], lr_scheduler_name) \
            if isinstance(lr_scheduler_name, str) else \
            [get_unique_attr_across([torch.optim.lr_scheduler, custom_lr_scheduler], ls) for ls in lr_scheduler_name]
        if isinstance(lr_scheduler_func, list):  # multiple lr_schedulers
            assert len(lr_scheduler_func) == len(lr_scheduler_params) == len(lr_scheduler_other_params), \
                "lr_scheduler, lr_scheduler_params and lr_scheduler_other_params lists must be of the same"
            assert len(optimizer) == len(lr_scheduler_func), \
                "when multiple lr_schedulers provided, number of optimizers must match number of lr_schedulers"
            loaded_lr_scheduler = [{**{'scheduler': ls(opt, **ls_p)}, **ls_op} for opt, ls, ls_p, ls_op
                                   in zip(optimizer, lr_scheduler_func, lr_scheduler_params, lr_scheduler_other_params)]
        else:  # one lr_scheduler
            if isinstance(optimizer, list):  # multiple optimizers
                loaded_lr_scheduler = [{**{'scheduler': lr_scheduler_func(opt, **lr_scheduler_params)},
                                      **lr_scheduler_other_params} for opt in optimizer]
            else:  # one optimizer
                loaded_lr_scheduler = {
                    'scheduler': lr_scheduler_func(optimizer, **lr_scheduler_params),
                    **lr_scheduler_other_params,  # monitor, interval, frequency, etc.
        }
    return loaded_lr_scheduler


if __name__ == '__main__':
    # example usage
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    lr_scheduler = load_lr_scheduler('StepLR', dict(step_size=5, gamma=0.1), dict(monitor='val_loss'), optimizer)
    print(lr_scheduler)

    optimizer = [torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
                 torch.optim.Adam(model.parameters(), lr=0.001)]
    lr_scheduler = load_lr_scheduler(['StepLR', 'ReduceLROnPlateau'],
                                     [dict(step_size=5, gamma=0.1), dict(mode='min', factor=0.1, patience=3)],
                                     [dict(monitor='val_loss'), dict(monitor='val_loss')],
                                     optimizer)
    print(lr_scheduler)
