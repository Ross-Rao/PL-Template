# python import
import logging
# package import
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix, MetricCollection
# local import

logger = logging.getLogger(__name__)
__all__ = ['ClassificationMetrics']

class ClassificationMetrics:
    def __init__(self, num_classes: int, average: str = 'macro', task: str = 'multiclass'):
        multi_cls_param = dict(average=average, task=task, num_classes=num_classes)
        metrics = {
            "accuracy": Accuracy(**multi_cls_param),
            "precision": Precision(**multi_cls_param),
            "recall": Recall(**multi_cls_param),
            "f1": F1Score(**multi_cls_param),
            "auc": AUROC(**multi_cls_param),
        }
        self.confusion_matrix = ConfusionMatrix(num_classes=num_classes, task='multiclass').eval()
        self.metrics = {'train': MetricCollection(metrics, prefix="train/"),
                        'val': MetricCollection(metrics, prefix="val/").eval(),
                        'test': MetricCollection(metrics, prefix="test/").eval()}

    def to(self, device):
        for v in self.metrics.values():
            v.to(device)

    def update(self, preds, target, stage, softmax=True):
        if preds.ndim == target.ndim:  # convert one-hot to index
            target = target.argmax(dim=-1)
        probs = preds.softmax(dim=-1) if softmax else preds
        self.metrics[stage].update(probs, target)
        if stage == 'test':
            self.confusion_matrix.update(preds, target)

    def compute_and_reset(self, stage):
        result_dict = {}
        for metric_name, metric in self.metrics[stage].items():
            if metric.update_count > 0:
                res = metric.compute()
                result_dict[metric_name] = res.item()
                metric.reset()
        if stage == 'test':
            if self.confusion_matrix.update_count > 0:
                result_dict['test/confusion_matrix'] = self.confusion_matrix.compute().cpu().numpy()
                self.confusion_matrix.reset()
        return result_dict


if __name__ == '__main__':
    import torch
    import numpy as np

    num_class = 3
    batch_size = 8
    num_batches = 5

    cls_metrics = ClassificationMetrics(num_classes=num_class, average='macro', task='multiclass')

    for _ in range(num_batches):
        y_hat = torch.randn(batch_size, num_class)
        y = torch.from_numpy(np.random.randint(0, num_class, size=(batch_size,)))
        cls_metrics.update(y_hat, y, stage='test')

    test_results = cls_metrics.compute_and_reset(stage='test')

    print("Test Results:", test_results)