# python import
# package import
import torch
import numpy as np
from monai.data import MetaTensor
from monai.transforms import MapTransform
# local import

__all__ = ['ExtractSlicesD']


class ExtractSlicesD(MapTransform):
    def __init__(self, keys, k, dim=0, flag=None):
        super().__init__(keys)
        self.k = k
        self.dim = dim
        self.flag = flag  # 可以是整数（起始索引）或浮点数（百分比）

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            original_type = type(image)

            # 将图像转换为 NumPy 数组进行处理
            if isinstance(image, MetaTensor) or isinstance(image, torch.Tensor):
                img_array = image.numpy()
            elif isinstance(image, np.ndarray):
                img_array = image
            else:
                raise ValueError(f"Unsupported input type: {original_type}")

            # 获取指定维度的切片数量
            num_slices = img_array.shape[self.dim]

            if isinstance(self.flag, int):
                # 如果 flag 是整数，则作为起始索引
                start_idx = self.flag
            elif isinstance(self.flag, float):
                # 如果 flag 是浮点数，则计算每个切片中像素值在前 flag 百分位的平均值
                # 并找到长度为 k 的切片窗口，其平均值之和最大
                if self.flag < 0 or self.flag > 1:
                    raise ValueError("浮点数 flag 必须在 0 到 1 之间")

                # 计算每个切片的平均值
                slice_values = []
                for i in range(num_slices):
                    # 提取切片
                    slice_data = np.take(img_array, [i], axis=self.dim)
                    # 计算切片中像素值在前 flag 百分位的平均值
                    slice_data_flat = slice_data.flatten()
                    threshold = np.percentile(slice_data_flat, 100 * self.flag)
                    top_pixels = slice_data_flat[slice_data_flat > threshold]
                    if len(top_pixels) == 0:
                        avg_value = 0
                    else:
                        avg_value = np.mean(top_pixels)
                    slice_values.append(avg_value)

                # 找到长度为 k 的切片窗口，其平均值之和最大
                max_sum = -1
                best_start = 0
                for i in range(num_slices - self.k + 1):
                    current_sum = sum(slice_values[i:i+self.k])
                    if current_sum > max_sum:
                        max_sum = current_sum
                        best_start = i
                start_idx = best_start
            else:
                # 如果 flag 是 None，则从中间开始
                start_idx = num_slices // 2 - self.k // 2

            end_idx = start_idx + self.k

            # 确保索引在有效范围内
            start_idx = max(0, start_idx)
            end_idx = min(num_slices, end_idx)

            # 提取切片
            if original_type == np.ndarray:
                # 对于 ndarray，直接切片
                d[key] = np.take(img_array, np.arange(start_idx, end_idx), axis=self.dim)
            elif original_type == torch.Tensor:
                # 对于 PyTorch Tensor，使用 torch 的切片操作
                d[key] = torch.take(torch.from_numpy(img_array), torch.arange(start_idx, end_idx), dim=self.dim)
            elif original_type == MetaTensor:
                # 对于 MetaTensor，切片并保留元数据
                sliced_image = np.take(img_array, np.arange(start_idx, end_idx), axis=self.dim)
                d[key] = MetaTensor(sliced_image, meta=image.meta)
            else:
                raise ValueError(f"Unsupported data type: {original_type}")

        return d