# python import
import logging
# package import
import torch
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError
# local import

logger = logging.getLogger(__name__)
__all__ = ['RegressionMetrics']

class RegressionMetrics:
    def __init__(self):
        metrics = {
            "recon_mae": MeanAbsoluteError(),
            "recon_mse": MeanSquaredError(),
        }
        self.metrics = {'train': MetricCollection(metrics, prefix="train/"),
                        'val': MetricCollection(metrics, prefix="val/").eval(),
                        'test': MetricCollection(metrics, prefix="test/").eval()}

    def update(self, preds, target, stage):
        self.metrics[stage].update(preds.reshape(-1), target.reshape(-1))

    def to(self, device):
        for v in self.metrics.values():
            v.to(device)

    def compute_and_reset(self, stage):
        result_dict = {}
        for metric_name, metric in self.metrics[stage].items():
            if metric.update_count > 0:
                res = metric.compute()
                result_dict[metric_name] = res.item()
                metric.reset()
        return result_dict


if __name__ == '__main__':
    import numpy as np

    batch_size = 8
    num_batches = 5

    reg_metrics = RegressionMetrics()

    for _ in range(num_batches):
        y_hat = torch.randn(batch_size)
        y = torch.from_numpy(np.random.randn(batch_size).astype(np.float32))
        reg_metrics.update(y_hat, y, stage='test')

    test_results = reg_metrics.compute_and_reset(stage='test')

    print("Test Results:", test_results)