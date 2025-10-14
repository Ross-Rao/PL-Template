# python import
import logging
from typing import Any, Union
# package import
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
# local import
from load_torch_model import load_model
from load_torch_optimizer import load_optimizer
from load_torch_lr_scheduler import load_lr_scheduler
from utils.cls_metrics import ClassificationMetrics
from utils.regression_metrics import RegressionMetrics
from utils.reconstruction_metrics import ReconstructionMetrics
from utils.generation_metrics import GenerationMetrics
# extra import
from copy import deepcopy
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


logger = logging.getLogger(__name__)
__all__ = ['TrainModule']


class TrainModule(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 model: Union[str, list[str]],
                 model_params: Union[dict, list[dict]],
                 optimizer: Union[str, list[str]],
                 optimizer_params: Union[dict, list[dict]],
                 criterion: Union[str, list[str]],
                 criterion_params: Union[dict, list[dict], None] = None,
                 lr_scheduler: Union[str, list[str], None] = None,
                 lr_scheduler_params: Union[dict, list[dict], None] = None,
                 lr_scheduler_other_params: Union[dict, list[dict], None] = None,
                 **kwargs):
        super().__init__()

        self.model = load_model(model, model_params)
        self.criterion = load_model(criterion, criterion_params)
        self.optimizer = load_optimizer(optimizer, optimizer_params, self.model)
        self.lr_scheduler = load_lr_scheduler(lr_scheduler, lr_scheduler_params, lr_scheduler_other_params, self.optimizer)

        # metrics
        self.cls_metrics = ClassificationMetrics(num_classes=num_classes)
        self.recon_metrics = ReconstructionMetrics()
        self.reg_metrics = RegressionMetrics()
        self.gen_metrics = GenerationMetrics(fid_feature=2048)
        # --------------------------------------------------------------------------------------- #
        # assert 0 and kwargs.keys(), "add extra config here, please check your code"
        self.n_steps = kwargs.get('n_steps')
        self.ema_model = AveragedModel(self.model[1], multi_avg_fn=get_ema_multi_avg_fn(kwargs.get('ema_decay')))
        self._original_state = None
        # --------------------------------------------------------------------------------------- #

    def _calculate_metrics(self, y_hat_tp, y_tp, stage):
        # ensure matched y and y_hat is same
        # zip will stop at the shortest length
        y_hat_tp = y_hat_tp if isinstance(y_hat_tp, tuple) else (y_hat_tp,)
        y_tp = y_tp if isinstance(y_tp, tuple) else (y_tp,)

        for y_hat, y in zip(y_hat_tp, y_tp):
            if y_hat.shape[1] == 1:  # Regression task
                self.reg_metrics.update(y_hat, y, stage)
            elif y_hat.shape[1] >= 2:  # Multi-class classification task
                self.cls_metrics.update(y_hat, y, stage)
            elif len(y_hat.shape) == 4:
                self.recon_metircs.update(y_hat, y, stage)  # Image reconstruction task
                self.gen_metrics.update(y_hat, y, stage)   # Image generation task
            else:
                raise ValueError("Invalid shape for y_hat.")

    def _log_metrics(self, stage):
        for metric in [self.recon_metrics, self.reg_metrics, self.cls_metrics]:
            res = metric.compute_and_reset(stage)
            cm = res.pop('test/confusion_matrix', None)
            if cm is not None:
                logger.info(f"Confusion Matrix:\n{cm}")
            if stage in ['val', 'test']:
                self.log_dict(res, prog_bar=True)
                for k, v in res.items():
                    logger.info(f"{k}: {v}")

    def configure_optimizers(self):
        """
        Set optimizer and lr_scheduler(optional)
        """
        optimizer, lr_scheduler = self.optimizer, self.lr_scheduler
        if lr_scheduler is not None:
            if isinstance(optimizer, list):
                return [optimizer[0]], [lr_scheduler[0]]
            return [optimizer], [lr_scheduler]
        else:
            if isinstance(optimizer, list):
                return optimizer[0]
            return optimizer

    def get_batch(self, batch):
        if isinstance(batch, list):
            return batch[0], batch[1]
        elif isinstance(batch, dict):
            # --------------------------------------------------------------------------------------- #
            # assert 0 and batch, "configure your input here, please check your code"
            image = batch['image'].as_tensor()
            image = image.reshape(-1, 1, *image.shape[2:])
            t = torch.randint(0, self.n_steps, (image.shape[0], 1)).to(image.device)
            eps = torch.randn_like(image).to(image.device)
            return (image, t, eps), eps
            # --------------------------------------------------------------------------------------- #
        else:
            raise ValueError('Invalid batch type')

    @staticmethod
    def get_batch_size(batch):
        if isinstance(batch, list):
            return batch[0].size(0)
        elif isinstance(batch, dict):
            return batch['index'].size(0)
        else:
            raise ValueError('Invalid batch type')

    def model_step(self, batch, batch_idx):
        x, y = self.get_batch(batch)
        model_params = x if isinstance(x, tuple) else (x,)
        # --------------------------------------------------------------------------------------- #
        # assert 0 and model_params, "execute your model with your input here, please check your code"
        # y_hat = self.model(*model_params)
        x_t = self.model[0].sample_forward(*model_params)
        y_hat = self.model[1](x_t, model_params[1])
        # --------------------------------------------------------------------------------------- #
        return y, y_hat


    def criterion_step(self, y, y_hat):
        criterion_params = (y_hat if isinstance(y_hat, tuple) else (y_hat,)) + (y if isinstance(y, tuple) else (y,))
        # --------------------------------------------------------------------------------------- #
        # assert 0 and criterion_params, "add your own code here to match the output with loss, please check your code"
        # be sure that y_hat params first and y params later in your criterion function
        loss = self.criterion(*criterion_params)
        # --------------------------------------------------------------------------------------- #
        return loss

    def training_step(self, batch, batch_idx):
        y, y_hat = self.model_step(batch, batch_idx)
        # self._calculate_metrics(y_hat, y, "train")  # not necessary, only debug
        loss = self.criterion_step(y, y_hat)
        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # log step loss
        loss_dt = {f'train/{k}': float(v) for k, v in outputs.items()}
        self.log_dict(loss_dt, batch_size=self.get_batch_size(batch), prog_bar=True)
        # --------------------------------------------------------------------------------------- #
        self.ema_model.update_parameters(self.model[1])
        # --------------------------------------------------------------------------------------- #

    def validation_step(self, batch, batch_idx):
        y, y_hat = self.model_step(batch, batch_idx)
        self._calculate_metrics(y_hat, y, "val")

        loss = self.criterion_step(y, y_hat)
        return loss

    def on_validation_batch_end(
            self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        # log step loss
        outputs = {'loss': outputs} if isinstance(outputs, torch.Tensor) else outputs
        loss_dt = {f'val/{k}': float(v) for k, v in outputs.items()}
        self.log_dict(loss_dt, batch_size=self.get_batch_size(batch), prog_bar=True)

    def test_step(self, batch, batch_idx):
        y, y_hat = self.model_step(batch, batch_idx)
        self._calculate_metrics(y_hat, y, "test")
        # --------------------------------------------------------------------------------------- #
        # ddpm generation
        with torch.no_grad():
            img = self.model[0].sample_backward((self.get_batch_size(batch), 1, 64, 64), self.model[1], simple_var=True)
            img = ((img + 1) / 2).reshape(-1, 1, *img.shape[2:])
            from torchvision.utils import make_grid
            grid = make_grid(img, nrow=7)
            self.logger.experiment.add_images(f"test_ddpm_img/{batch_idx}", grid,
                                              self.current_epoch, dataformats='CHW')
        # --------------------------------------------------------------------------------------- #

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train/loss')
        if train_loss is not None:
            self.log('train_loss', train_loss)  # train_loss is the key for callback
            logger.info(f"\nEpoch {self.current_epoch} - train_loss: {train_loss}")  # print train loss to log file
        self._log_metrics("train")  # not necessary, only debug

    # --------------------------------------------------------------------------------------- #
    def on_validation_epoch_start(self):
        torch.optim.swa_utils.update_bn(self.trainer.train_dataloader, self.ema_model)
        self._original_state = deepcopy(self.model[1].state_dict())
        self.model[1].load_state_dict(self.ema_model.module.state_dict())
    # --------------------------------------------------------------------------------------- #

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val/loss')
        if val_loss is not None:
            self.log('val_loss', val_loss)  # val_loss is the key for callback
            logger.info(f"\nEpoch {self.current_epoch} - val_loss: {val_loss}")  # print val loss to log file
        self._log_metrics("val")
        # --------------------------------------------------------------------------------------- #
        self.model[1].load_state_dict(self._original_state)
        # --------------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------------- #
    def on_test_epoch_start(self):
        torch.optim.swa_utils.update_bn(self.trainer.train_dataloader, self.ema_model)
        self._original_state = deepcopy(self.model[1].state_dict())
        self.model[1].load_state_dict(self.ema_model.module.state_dict())
    # --------------------------------------------------------------------------------------- #

    def on_test_epoch_end(self):
        self._log_metrics("test")
        # --------------------------------------------------------------------------------------- #
        self.model[1].load_state_dict(self._original_state)
        # --------------------------------------------------------------------------------------- #
