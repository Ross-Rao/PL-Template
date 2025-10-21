# pip install openslide-bin
# pip install openslide-python
from typing import Hashable, Sequence, Union
import numpy as np
import scipy.sparse as sp
from monai.data import WSIReader
from monai.transforms import Transform
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from torch_geometric.utils import from_scipy_sparse_matrix
from custom.models.pytorch_models import ResnetMLP
import logging

logger = logging.getLogger(__name__)
__all__ = ["CropPatchFromImageD", "BuildGraphD"]


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
        d = data
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


def update_adj(adj, cluster_labels, patch_embeddings, old_labels=None):
    # Get the unique cluster labels
    unique_cluster_labels = np.unique(cluster_labels)

    centroid_spots = []

    # For each cluster, find the spots closest to the centroid
    for cluster_label in unique_cluster_labels:
        # Find the spots in the cluster
        cluster_spots = np.where(cluster_labels == cluster_label)[0]

        # Find the spot closest to the centroid of the cluster
        if unique_cluster_labels.shape[0] == 1:
            # Pick a random spot as the centroid
            nearest_spot_idx = np.random.randint(0, len(cluster_spots))
            nearest_spot = cluster_spots[nearest_spot_idx]

        cluster_centroid = patch_embeddings[cluster_spots].mean(axis=0)
        # Find the nearest spot to the centroid
        nearest_spot_idx = np.argmin(np.linalg.norm(patch_embeddings[cluster_spots] - cluster_centroid, axis=1))
        nearest_spot = cluster_spots[nearest_spot_idx]

        # Connect the nearest spot to all other spots in the cluster
        for j in range(len(cluster_spots)):
            if cluster_spots[j] != nearest_spot:
                adj[cluster_spots[j], nearest_spot] = 1
                adj[nearest_spot, cluster_spots[j]] = 1

        # Save the nearest spot
        centroid_spots.append(nearest_spot_idx)

        # Multiply the cluster label of the nearest spot by -1
        cluster_labels[cluster_spots[nearest_spot_idx]] *= -1

        # If the cluster label is zero, set it to -(len+1)
        if cluster_labels[cluster_spots[nearest_spot_idx]] == 0:
            cluster_labels[cluster_spots[nearest_spot_idx]] = -(len(unique_cluster_labels))

    # Iterate over the old_labels, take the negative labels and append them to the centroid_spots
    if old_labels is not None:
        for j, old_label in enumerate(old_labels):
            if old_label < 0:
                centroid_spots.append(j)

                # Make the centroid_spots unique
    centroid_spots = list(set(centroid_spots))

    # Connect the centroid spots to each other
    for j in range(len(centroid_spots)):
        for k in range(j+1, len(centroid_spots)):
            adj[centroid_spots[j], centroid_spots[k]] = 1
            adj[centroid_spots[k], centroid_spots[j]] = 1

    return adj, cluster_labels


class BuildGraphD(Transform):
    def __init__(self, x_key, y_key, patch_key, model_path, edge_idx_key, clus_label_key, emb_key, n_cluster, max_iter, n_init):
        self.x_key = x_key
        self.y_key = y_key
        self.patch_key = patch_key
        self.edge_idx_key = edge_idx_key
        self.clus_label_key = clus_label_key
        self.feature_extractor = ResnetMLP(path=model_path, train=False)
        self.feature_extractor.eval()
        self.emb_key = emb_key
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.n_init = n_init

    def __call__(self, data: dict):
        d = data
        x_coords = d[self.x_key].squeeze().numpy()
        y_coords = d[self.y_key].squeeze().numpy()
        
        # one hop graph
        pts = np.column_stack([x_coords, y_coords])  # (N, 2)
        adj = kneighbors_graph(pts, n_neighbors=min(8, len(pts)-1), mode='connectivity', include_self=False).toarray()
        adj = adj + adj.T  # 对称邻接矩阵
        adj = (adj > 0).astype(np.float32)  # 二值化
        np.fill_diagonal(adj, 1)  # 自环

        # herarchical graph with clustering
        patches = d[self.patch_key]
        d[self.emb_key] = self.feature_extractor(patches)
        features = d[self.emb_key].numpy()  # (N, D)
        coords_clusterer = KMeans(n_clusters=self.n_cluster, max_iter=self.max_iter, n_init=self.n_init)
        coords_clusterer.fit(pts)
        d[self.clus_label_key] = coords_clusterer.predict(pts)
        adj, d[self.clus_label_key] = update_adj(adj, d[self.clus_label_key], features)
        spatial_clusterer = KMeans(n_clusters=self.n_cluster, max_iter=self.max_iter, n_init=self.n_init)
        spatial_clusterer.fit(features)
        second_cluster_label = spatial_clusterer.predict(features)
        adj, d[self.clus_label_key] = update_adj(adj, d[self.clus_label_key], features,
            old_labels=second_cluster_label)

        # convert to edge_index
        sp_adj = sp.coo_matrix(adj)
        edge_index, _ = from_scipy_sparse_matrix(sp_adj)
        d[self.edge_idx_key] = edge_index

        logger.info(f"Built graph with {edge_index.size(1)} edges for {len(patches)} patches. "
                    f"using CacheDataset will speed up the process and won't transform after epoch > 1.")
        return d