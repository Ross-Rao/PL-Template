# python import
import logging
from typing import Any, Union
# package import
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import (Accuracy, Precision, Recall, F1Score, AUROC, MeanAbsoluteError, MeanSquaredError,
                          ConfusionMatrix, MetricCollection)
from torchmetrics.image import (PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
                                LearnedPerceptualImagePatchSimilarity)
from torchmetrics.utilities.plot import plot_confusion_matrix
# local import
from custom import models as custom_models
from custom import lr_schedulers as custom_lr_scheduler
from utils.load_module import get_unique_attr_across

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
        # model structure settings
        assert isinstance(model, list) == isinstance(model_params, list), \
            "model and model_params must either both be lists or neither be lists"
        self.model = getattr(custom_models, model)(**model_params) \
            if isinstance(model_params, dict) and isinstance(model, str) else \
            torch.nn.ModuleList([getattr(custom_models, m)(**mp).to(self.device) for m, mp in zip(model, model_params)])

        # optimizer settings
        assert isinstance(optimizer, list) == isinstance(optimizer_params, list), \
            "optimizer and optimizer_params must either both be lists or neither be lists"
        def read_optimizer_params(md, dt):
            lr_list, params_name_list, model_list = dt['lr'], dt['params'], dt.get('model_id', None)
            extra_dt = {k: v for k, v in dt.items() if k not in ['lr', 'params', 'model_id']}
            return [{'params': getattr(md if model_id is None else md[model_id], name).parameters()
                    if hasattr(getattr(md if model_id is None else md[model_id], name), 'parameters')
                    else getattr(md if model_id is None else md[model_id], name),
                     'lr': lr, **extra_dt}
                    for lr, name, model_id in zip(lr_list, params_name_list, model_list)]

        if isinstance(optimizer, str) and isinstance(optimizer_params, dict):
            if optimizer_params.get('params', None) is None:
                self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), **optimizer_params)
            else:
                optimizer_params_ls = read_optimizer_params(self.model, optimizer_params)
                self.optimizer = getattr(torch.optim, optimizer)(optimizer_params_ls)
        else:
            assert all('params' in opt_p and 'model_id' in opt_p for opt_p in optimizer_params), \
                "Each optimizer_params dict must contain 'params' and 'model_id' keys"
            self.optimizer = [
                getattr(torch.optim, opt)(read_optimizer_params(self.model, opt_p))
                for opt, opt_p in zip(optimizer, optimizer_params)
            ]

        # loss function settings
        self.criterion = get_unique_attr_across([torch.nn, custom_models], {criterion: criterion_params})[0] \
            if isinstance(criterion, str) else \
            torch.nn.ModuleList(
                [get_unique_attr_across([torch.nn, custom_models], {c: cp})[0]
                 for c, cp in zip(criterion, criterion_params)]
            )

        # lr_scheduler settings
        lr_lt = [lr_scheduler, lr_scheduler_params, lr_scheduler_other_params]
        assert all(var is None for var in lr_lt) or all(var is not None for var in lr_lt), \
            'if lr_scheduler is valid, lr_scheduler_params and lr_scheduler_other_params must be provided'
        if lr_scheduler is None:
            self.lr_scheduler = None
        else:
            # StepLR, ReduceLROnPlateau, etc.
            lr_scheduler_func = get_unique_attr_across([torch.optim.lr_scheduler, custom_lr_scheduler], lr_scheduler) \
                if isinstance(lr_scheduler, str) else \
                [get_unique_attr_across([torch.optim.lr_scheduler, custom_lr_scheduler], ls) for ls in lr_scheduler]
            if isinstance(lr_scheduler_func, list):
                self.lr_scheduler = [{**{'scheduler': ls(opt, **ls_p)}, **ls_op}
                                     for opt, ls, ls_p, ls_op in zip(self.optimizer, lr_scheduler_func,
                                                                     lr_scheduler_params, lr_scheduler_other_params)]
            else:
                if isinstance(self.optimizer, list):
                    self.lr_scheduler = [{**{'scheduler': lr_scheduler_func(opt, **lr_scheduler_params)},
                                          **lr_scheduler_other_params}
                                         for opt in self.optimizer]
                else:
                    self.lr_scheduler = {
                        'scheduler': lr_scheduler_func(self.optimizer, **lr_scheduler_params),
                        **lr_scheduler_other_params,  # monitor, interval, frequency, etc.
                    }

        # metrics
        multi_cls_param = dict(average='macro', task='multiclass', num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(num_classes=num_classes, task='multiclass').eval()
        self._train_cls_metrics = MetricCollection({
            "accuracy": Accuracy(**multi_cls_param),
            "precision": Precision(**multi_cls_param),
            "recall": Recall(**multi_cls_param),
            "f1": F1Score(**multi_cls_param),
            "auc": AUROC(**multi_cls_param),
        }, prefix="train/")
        self._train_recon_metrics = MetricCollection({
            "psnr": PeakSignalNoiseRatio(),
            "ssim": StructuralSimilarityIndexMeasure(),
            "lpips": LearnedPerceptualImagePatchSimilarity(),
            "recon_mae": MeanAbsoluteError(),
            "recon_mse": MeanSquaredError(),
        }, prefix="train/")
        self._train_reg_metrics = MetricCollection({
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
        }, prefix="train/")
        self._val_cls_metrics = self._train_cls_metrics.clone(prefix="val/").eval()
        self._val_recon_metrics = self._train_recon_metrics.clone(prefix="val/").eval()
        self._val_reg_metrics = self._train_reg_metrics.clone(prefix="val/").eval()
        self._test_cls_metrics = self._train_cls_metrics.clone(prefix="test/").eval()
        self._test_recon_metrics = self._train_recon_metrics.clone(prefix="test/").eval()
        self._test_reg_metrics = self._train_reg_metrics.clone(prefix="test/").eval()
        self.cls_metrics = {
            'train': self._train_cls_metrics,
            'val': self._val_cls_metrics,
            'test': self._test_cls_metrics,
        }
        self.recon_metrics = {
            'train': self._train_recon_metrics,
            'val': self._val_recon_metrics,
            'test': self._test_recon_metrics,
        }
        self.reg_metrics = {
            'train': self._train_reg_metrics,
            'val': self._val_reg_metrics,
            'test': self._test_reg_metrics,
        }
        # --------------------------------------------------------------------------------------- #
        # assert 0 and kwargs.keys(), "add extra config here, please check your code"
        self.latent_dim = kwargs.get('latent_dim', 128)
        self.criterion.pareto.share_model = self.model.encoder
        # --------------------------------------------------------------------------------------- #

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

    @staticmethod
    def perturbate_embedding(B: int, latent_dim: int, device: torch.device = torch.device('cpu')):
        """
        B: batch size
        latent_dim: 维度总数
        device: 输出设备
        返回: pert_vec, GT_vec  (B, latent_dim) 均为 torch.float32
        """
        pert_vec = torch.zeros((B, latent_dim), dtype=torch.float32, device=device)
        GT_vec   = torch.zeros((B, latent_dim), dtype=torch.float32, device=device)

        values = torch.linspace(-1.5, 1.5, 16, dtype=torch.float32, device=device)

        for i in range(B):
            dim = torch.randint(latent_dim, (1,)).item()
            val = values[torch.randint(len(values), (1,)).item()]
            pert_vec[i, dim] = val
            GT_vec[i, dim]   = 1

        return pert_vec, GT_vec

    def get_batch(self, batch):
        if isinstance(batch, list):
            return batch[0], batch[1]
        elif isinstance(batch, dict):
            # --------------------------------------------------------------------------------------- #
            # assert 0, "configure your input here, please check your code"
            pert_vec, gt = self.perturbate_embedding(
                batch['image'].size(0), self.latent_dim, device=batch['image'].device)
            return (batch['image'].as_tensor(), pert_vec), (batch['image'].as_tensor(), gt)
            # --------------------------------------------------------------------------------------- #
        else:
            raise ValueError('Invalid batch type')

    @staticmethod
    def get_batch_size(batch):
        if isinstance(batch, list):
            return batch[0].size(0)
        elif isinstance(batch, dict):
            return batch['image'].size(0)
        else:
            raise ValueError('Invalid batch type')

    def model_step(self, batch, batch_idx):
        x, y = self.get_batch(batch)
        model_params = x if isinstance(x, tuple) else (x,)
        # --------------------------------------------------------------------------------------- #
        # assert 0, "execute your model with your input here, please check your code"
        y_hat = self.model(*model_params)
        # --------------------------------------------------------------------------------------- #
        return y, y_hat


    def criterion_step(self, y, y_hat):
        criterion_params = (y_hat if isinstance(y_hat, tuple) else (y_hat,)) + (y if isinstance(y, tuple) else (y,))
        # --------------------------------------------------------------------------------------- #
        # assert 0, "add your own code here to match the output with loss, please check your code"
        # be sure that y_hat params first and y params later in your criterion function
        loss = self.criterion(*criterion_params)
        # --------------------------------------------------------------------------------------- #
        return loss

    def training_step(self, batch, batch_idx):
        y, y_hat = self.model_step(batch, batch_idx)
        # self._update_metrics(y_hat, y, "train")  # not necessary, only debug
        loss = self.criterion_step(y, y_hat)
        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # log step loss
        loss_dt = {f'train/{k}': float(v) for k, v in outputs.items()}
        self.log_dict(loss_dt, batch_size=self.get_batch_size(batch), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        y, y_hat = self.model_step(batch, batch_idx)
        self._update_metrics(y_hat[1:3], (y[0], batch['label'].as_tensor()), "val")

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
        self._update_metrics(y_hat[1:3], (y[0], batch['label'].as_tensor()), "test")
        # explain part start here
        recon_img, super_img = y_hat[0], y_hat[1]
        # log a batch of images
        self.logger.experiment.add_images(f"test_recon_img/{batch_idx}",
                                          recon_img.reshape(-1, 1, *recon_img.shape[2:]), self.current_epoch,
                                          dataformats='NCHW')
        self.logger.experiment.add_images(f"test_super_img/{batch_idx}",
                                          super_img.reshape(-1, 1, *super_img.shape[2:]), self.current_epoch,
                                          dataformats='NCHW')
        raw_img = y[0]
        pert_vec = torch.cat([torch.eye(14, device=raw_img.device), torch.zeros(14, 350 - 14, device=raw_img.device)], dim=1)
        pert_vec = pert_vec.repeat(raw_img.shape[0], 1) * 1
        raw_img = torch.repeat_interleave(raw_img, repeats=14, dim=0)
        pert_img = self.model(raw_img, pert_vec)[-1] - raw_img
        def tensor_to_cmap_mask(tensor_4d, raw_img_4d, alpha=0.5, cmap='jet', nrow=14):
            """
            tensor_4d : (N,1,H,W)  权重/响应图  0-1
            raw_img_4d: (N,1,H,W)  原图 灰度     0-1
            alpha     : 热力图透明度
            return    : (3,H*,W*)  叠加图 RGB  0-1
            """
            import matplotlib.cm as cm
            from torchvision.utils import make_grid

            # 1. 权重图 cmap → (N,H,W,3)
            weight_3d = tensor_4d.squeeze(1).cpu()
            w_min = weight_3d.amin(dim=(1, 2), keepdim=True)
            w_max = weight_3d.amax(dim=(1, 2), keepdim=True)
            weight_3d = (weight_3d - w_min) / (w_max - w_min + 1e-8)  # (N,H,W)
            colored = cm.get_cmap(cmap)(weight_3d.numpy())[..., :3]   # (N,H,W,3)
            colored = torch.from_numpy(colored).float()

            # 2. 原图灰度 → RGB（复制通道）
            raw_gray = raw_img_4d.squeeze(1).cpu()                   # (N,H,W)
            raw_rgb  = raw_gray.unsqueeze(-1).expand(-1, -1, -1, 3)  # (N,H,W,3)

            # 4. 加权混合
            blended = alpha * colored + (1 - alpha) * raw_rgb
            blended = blended.clamp(0, 1)
            return make_grid(blended.permute(0, 3, 1, 2), nrow=nrow)

        self.logger.experiment.add_images(f"test_pert_img/{batch_idx}",
                                          tensor_to_cmap_mask(pert_img.reshape(-1, 1, *pert_img.shape[2:]),
                                                              raw_img.reshape(-1, 1, *raw_img.shape[2:])),
                                          self.current_epoch, dataformats='CHW')

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train/loss')
        if train_loss is not None:
            self.log('train_loss', train_loss)  # train_loss is the key for callback
            logger.info(f"\nEpoch {self.current_epoch} - train_loss: {train_loss}")  # print train loss to log file
        # not necessary, only debug
        for metrics_dict in [self.cls_metrics['train'], self.reg_metrics['train'], self.recon_metrics['train']]:
            for metric_name, metric in metrics_dict.items():
                # re-comment `update_metrics` in training_step if you want to use this
                if metric.update_count > 0:
                    res = metric.compute()
                    self.log(metric_name, res, prog_bar=True)
                    metric.reset()

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val/loss')
        if val_loss is not None:
            self.log('val_loss', val_loss)  # val_loss is the key for callback
            logger.info(f"\nEpoch {self.current_epoch} - val_loss: {val_loss}")  # print val loss to log file
        for metrics_dict in [self.cls_metrics['val'], self.reg_metrics['val'], self.recon_metrics['val']]:
            for metric_name, metric in metrics_dict.items():
                if metric.update_count > 0:
                    res = metric.compute()
                    self.log(metric_name, res, prog_bar=True)
                    logger.info(f"Epoch {self.current_epoch} - {metric_name}: {res}")
                    metric.reset()

    def on_test_epoch_end(self):
        if self.confusion_matrix.update_count > 0:
            cm = self.confusion_matrix.compute()
            plt, _ = plot_confusion_matrix(cm)
            self.logger.experiment.add_figure(f"test_confusion_matrix", plt)
            logger.info(f"Confusion Matrix:\n{cm}")
            self.confusion_matrix.reset()
        for metrics_dict in [self.cls_metrics['test'], self.reg_metrics['test'], self.recon_metrics['test']]:
            for metric_name, metric in metrics_dict.items():
                if metric.update_count > 0:
                    res = metric.compute()
                    self.log(metric_name, res, prog_bar=True)
                    logger.info(f"{metric_name}: {res}")
                    metric.reset()

    def _update_metrics(self, y_hat_tp, y_tp, stage):
        # ensure matched y and y_hat is same
        # zip will stop at the shortest length
        y_hat_tp = y_hat_tp if isinstance(y_hat_tp, tuple) else (y_hat_tp,)
        y_tp = y_tp if isinstance(y_tp, tuple) else (y_tp,)

        for y_hat, y in zip(y_hat_tp, y_tp):
            if len(y_hat.shape) == 2 or len(y_hat.shape) == 3:
                if len(y_hat.shape) == 3:
                    y_hat = y_hat.reshape(-1, y_hat.shape[-1])
                    y = y.unsqueeze(1).repeat(1, 3).reshape(-1)
                if y_hat.shape[1] == 1:  # Regression task
                    self._regression_metrics(y_hat, y, stage)
                elif y_hat.shape[1] >= 2:  # Multi-class classification task
                    if y_hat.ndim == y.ndim:  # convert one-hot to index
                        y = y.argmax(dim=-1)
                    self._multiclass_classification_metrics(y_hat, y, stage)
                    if stage == "test":
                        self.confusion_matrix.update(y_hat, y)
                else:
                    raise ValueError("Invalid shape for y_hat.")
            elif len(y_hat.shape) == 4:  # Image reconstruction task
                self._image_reconstruction_metrics(y_hat, y, stage)

    def _regression_metrics(self, y_hat, y, stage):
        self.reg_metrics[stage].update(y_hat.reshape(-1), y)

    def _multiclass_classification_metrics(self, y_hat, y, stage):
        # --------------------------------------------------------------------------------------- #
        # assert 0, "your model only needs to return logits, please check your code"
        probs = torch.softmax(y_hat, dim=1)
        # --------------------------------------------------------------------------------------- #
        self.cls_metrics[stage].update(probs, y)

    def _image_reconstruction_metrics(self, y_hat, y, stage):
        # input range is [0, 1], adjust to [-1, 1] for LPIPS
        y_hat_lpips = y_hat.repeat(1, 3, 1, 1) * 2 - 1 if y_hat.size(1) == 1 else y_hat * 2 - 1
        y_lpips = y.repeat(1, 3, 1, 1) * 2 - 1 if y.size(1) == 1 else y * 2 - 1
        y_hat_lpips = torch.clamp(y_hat_lpips, -1, 1)

        # 更新其他指标
        for metric_name, metric in self.recon_metrics[stage].items():
            if metric_name != f"{stage}/lpips":
                metric.update(y_hat, y)
        # 单独更新lpips
        self.recon_metrics[stage]["lpips"].update(y_hat_lpips, y_lpips)
