# pip install openslide-bin
# pip install openslide-python
from typing import Hashable, Sequence, Union
import numpy as np
import torch
from monai.data import WSIReader
from monai.transforms import Transform

__all__ = ["CropPatchFromImageD"]


class CropPatchFromImageD(Transform):
    """
    Dictionary-based transform.
    从 WSI 文件路径中按 (x, y) 抠 patch，返回 (H, W, C) np.ndarray。
    支持多分辨率：level=0 为最高分辨率。
    """

    def __init__(
            self,
            patch_size: Union[Sequence[int], int] = 256,
            image_key: Hashable = "image",  # 这里传的是 WSI 文件路径
            x_key: Hashable = "x",
            y_key: Hashable = "y",
            patch_key: Hashable = "patches",
            level: int = 0,  # 0=最高分辨率
            backend: str = "openslide",  # 或 "tifffile"
    ):
        self.patch_size = np.atleast_1d(patch_size).astype(int)
        if self.patch_size.size == 1:
            self.patch_size = np.repeat(self.patch_size, 2)
        self.image_key = image_key
        self.x_key = x_key
        self.y_key = y_key
        self.patch_key = patch_key
        self.level = level
        self.reader = WSIReader(backend=backend)

    def _crop_one(self, xc: float, yc: float, path: str):
        half = self.patch_size // 2
        x0 = int(round(xc)) - half[1]
        y0 = int(round(yc)) - half[0]

        # 1. 打开 WSI（返回 OpenSlide 对象）
        wsi_obj = self.reader.read(path)

        # 2. 用 reader.get_data 提取区域
        patch, _ = self.reader.get_data(
            wsi_obj,
            location=(x0, y0),
            size=tuple(self.patch_size),
            level=self.level,
        )
        # patch 形状 (C, H, W) -> (H, W, C)
        return patch.transpose(1, 2, 0)   # -> (H, W, C)

    def __call__(self, data: dict):
        d = dict(data)
        path = d[self.image_key]  # WSI 文件路径
        xc = d[self.x_key]
        yc = d[self.y_key]
        # 统一成列表
        xc_list = np.atleast_1d(xc)
        yc_list = np.atleast_1d(yc)

        half = self.patch_size // 2
        patches = []
        for xc_i, yc_i in zip(xc_list, yc_list):
            x0 = int(round(xc_i)) - half[1]
            y0 = int(round(yc_i)) - half[0]
            patches.append(self._crop_one(x0, y0, path))

        # 沿新轴堆叠: (N, H, W, C)
        d[self.patch_key] = np.stack(patches, axis=0)
        return d