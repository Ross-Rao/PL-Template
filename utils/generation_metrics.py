# python import
import logging
# package import
import torch
from torchmetrics import MetricCollection
from torchmetrics.image import FrechetInceptionDistance
# local import

logger = logging.getLogger(__name__)
__all__ = ['GenerationMetrics']

class GenerationMetrics:
    def __init__(self, fid_feature=2048):
        metrics = {
            "fid": FrechetInceptionDistance(feature=fid_feature),
        }
        self.metrics = {'train': MetricCollection(metrics, prefix="train/"),
                        'val': MetricCollection(metrics, prefix="val/").eval(),
                        'test': MetricCollection(metrics, prefix="test/").eval()}

    def update(self, preds, target, stage):
        for metric_name, metric in self.metrics[stage].items():
            if metric_name == f"{stage}/fid":
                y_hat_fid = y_hat.repeat(1, 3, 1, 1) if y_hat.size(1) == 1 else y_hat
                y_fid = y.repeat(1, 3, 1, 1) if y.size(1) == 1 else y
                y_hat_fid = (((y_hat_fid + 1) / 2).clamp(0, 1) * 255).byte() if y_hat_fid.dtype != torch.uint8 else y_hat_fid
                y_fid = (((y_fid + 1) / 2).clamp(0, 1) * 255).byte() if y_fid.dtype != torch.uint8 else y_fid
                self.metrics[stage]["fid"].update(y_hat_fid, real=False)
                self.metrics[stage]["fid"].update(y_fid, real=True)
            else:
                metric.update(preds, target)

    def compute_and_reset(self, stage):
        result_dict = {}
        for metric_name, metric in self.metrics[stage].items():
            if metric.update_count > 0:
                res = metric.compute()
                result_dict[metric_name] = res
                metric.reset()
        return result_dict


if __name__ == '__main__':
    batch_size = 8
    num_batches = 5
    img_channels = 1
    img_height = 224
    img_width = 224

    recon_metrics = GenerationMetrics()

    for _ in range(num_batches):
        y_hat = torch.rand(batch_size, img_channels, img_height, img_width)
        y = torch.rand(batch_size, img_channels, img_height, img_width)
        recon_metrics.update(y_hat, y, stage='test')

    test_results = recon_metrics.compute_and_reset(stage='test')

    print("Test Results:", test_results)