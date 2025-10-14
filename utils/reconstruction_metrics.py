# python import
import logging
# package import
import torch
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError
from torchmetrics.image import (PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
                                LearnedPerceptualImagePatchSimilarity)
# local import

logger = logging.getLogger(__name__)
__all__ = ['ReconstructionMetrics']

class ReconstructionMetrics:
    def __init__(self):
        metrics = {
            "psnr": PeakSignalNoiseRatio(),
            "ssim": StructuralSimilarityIndexMeasure(),
            "lpips": LearnedPerceptualImagePatchSimilarity(),
            "recon_mae": MeanAbsoluteError(),
            "recon_mse": MeanSquaredError(),
        }
        self.metrics = {'train': MetricCollection(metrics, prefix="train/"),
                        'val': MetricCollection(metrics, prefix="val/").eval(),
                        'test': MetricCollection(metrics, prefix="test/").eval()}

    def to(self, device):
        for v in self.metrics.values():
            v.to(device)

    def update(self, preds, target, stage):
        for metric_name, metric in self.metrics[stage].items():
            if metric_name == f"{stage}/lpips":
                # input range is [0, 1], adjust to [-1, 1] for LPIPS
                y_hat_lpips = preds.repeat(1, 3, 1, 1) * 2 - 1 if preds.size(1) == 1 else preds * 2 - 1
                y_lpips = target.repeat(1, 3, 1, 1) * 2 - 1 if target.size(1) == 1 else target * 2 - 1
                y_hat_lpips = torch.clamp(y_hat_lpips, -1, 1)
                self.metrics[stage]["lpips"].update(y_hat_lpips, y_lpips)
            else:
                metric.update(preds, target)

    def compute_and_reset(self, stage):
        result_dict = {}
        for metric_name, metric in self.metrics[stage].items():
            if metric.update_count > 0:
                res = metric.compute()
                result_dict[metric_name] = res.item()
                metric.reset()
        return result_dict


if __name__ == '__main__':
    batch_size = 8
    num_batches = 5
    img_channels = 1
    img_height = 224
    img_width = 224

    recon_metrics = ReconstructionMetrics()

    for _ in range(num_batches):
        y_hat = torch.rand(batch_size, img_channels, img_height, img_width)
        y = torch.rand(batch_size, img_channels, img_height, img_width)
        recon_metrics.update(y_hat, y, stage='test')

    test_results = recon_metrics.compute_and_reset(stage='test')

    print("Test Results:", test_results)