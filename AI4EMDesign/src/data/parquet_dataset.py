# dataset_em_parquet.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import logging
import torch
from functools import partial
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from torch.utils import data
import torchvision.transforms as T
from typing import Tuple, Optional, Callable

from src.registry import DATA

_logger = logging.getLogger(__name__)

META_SHAPE_SUFFIX = "_shape"
META_DTYPE_SUFFIX = "_dtype"


def scale_to_minus1_1(x):
    return x * 2.0 - 1.0

def downsample_s_curve(x, factor):
    return x[::factor]


def normalize_s_curve_with_stats(x, mean, max_value, min_value):
    return (x - mean) / (max_value - min_value)


def _freq_to_index(freq, start_freq, end_freq, num_points):
    return int(round((freq - start_freq) * (num_points - 1) / (end_freq - start_freq)))


def frequency_select_s_curve_by_freq(
    x,
    original_start_freq,
    original_end_freq,
    original_num_points,
    crop_start_freq,
    crop_end_freq,
):
    left = _freq_to_index(
        crop_start_freq,
        original_start_freq,
        original_end_freq,
        original_num_points,
    )
    right = _freq_to_index(
        crop_end_freq,
        original_start_freq,
        original_end_freq,
        original_num_points,
    ) + 1
    return x[..., left:right]


def get_x_transforms(enable_scale_to_minus1_1, enable_unsqueeze_channel=False):
    ops = []
    if enable_unsqueeze_channel:
        ops.append(T.Lambda(lambda x: x.unsqueeze(0)))
    if enable_scale_to_minus1_1:
        ops.append(scale_to_minus1_1)
    return T.Compose(ops) if ops else None


def get_y_transforms(
    downsample_factor,
    enable_frequency_crop=False,
    original_start_freq=2.0,
    original_end_freq=20.0,
    original_num_points=1000,
    crop_start_freq=None,
    crop_end_freq=None,
    enable_stat_normalize=False,
    normalize_mean=0.0,
    normalize_max=1.0,
    normalize_min=0.0,
):
    ops = []
    if enable_frequency_crop:
        ops.append(
            partial(
                frequency_select_s_curve_by_freq,
                original_start_freq=original_start_freq,
                original_end_freq=original_end_freq,
                original_num_points=original_num_points,
                crop_start_freq=crop_start_freq,
                crop_end_freq=crop_end_freq,
            )
        )
    if downsample_factor > 1:
        ops.append(partial(downsample_s_curve, factor=downsample_factor))
    if enable_stat_normalize:
        ops.append(
            partial(
                normalize_s_curve_with_stats,
                mean=normalize_mean,
                max_value=normalize_max,
                min_value=normalize_min,
            )
        )
    return T.Compose(ops) if ops else None


def _parse_shape_and_dtype(meta: dict, key: str) -> Tuple[Tuple[int, ...], np.dtype]:
    shape_expr = meta[f"{key}{META_SHAPE_SUFFIX}"]
    dtype_expr = meta[f"{key}{META_DTYPE_SUFFIX}"]
    shape = tuple(eval(shape_expr))
    dtype = np.dtype(eval(f"np.{dtype_expr}"))
    return shape, dtype


def _to_array_by_metadata(x, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist(), dtype=dtype)
    else:
        arr = arr.astype(dtype, copy=False)
    return arr.reshape(shape).copy()


def _build_reconstruct_transform(
    shape: Tuple[int, ...],
    np_dtype: np.dtype,
):
    def _fn(x):
        arr = _to_array_by_metadata(x, shape=shape, dtype=np_dtype)
        return torch.from_numpy(arr)
    return _fn


class EMTopologySCurveParquetDataset(Dataset):
    """
    PyTorch 风格的数据集类，用于读取电磁拓扑 + S 曲线的 Parquet 数据集。

    Parquet 文件中每一行包含：
    - topology : 4x4 的二值矩阵（嵌套 list）
    - s_curve  : 长度为 1000 的 S 曲线

    __getitem__ 返回：
    - topology_tensor : (4,4) 或 (1,4,4)
    - s_curve_tensor  : (1000,)
    """

    def __init__(
        self,
        parquet_path: str,
        x_key: str,
        y_key: str,
        x_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        y_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()

        self.parquet_path = parquet_path
        self.x_key = x_key
        self.y_key = y_key
        self.x_transform = x_transform
        self.y_transform = y_transform

        # 读取 Parquet 文件
        self.df = pd.read_parquet(self.parquet_path)
        schema = pq.read_schema(self.parquet_path)
        raw_meta = schema.metadata or {}
        self.meta = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in raw_meta.items()
        }

        x_shape, x_np_dtype = _parse_shape_and_dtype(self.meta, self.x_key)
        y_shape, y_np_dtype = _parse_shape_and_dtype(self.meta, self.y_key)

        x_base_transform = _build_reconstruct_transform(
            shape=x_shape,
            np_dtype=x_np_dtype,
        )
        y_base_transform = _build_reconstruct_transform(
            shape=y_shape,
            np_dtype=y_np_dtype,
        )

        self.x_transform = T.Compose(
            [x_base_transform, self.x_transform]
            if self.x_transform is not None else [x_base_transform]
        )
        self.y_transform = T.Compose(
            [y_base_transform, self.y_transform]
            if self.y_transform is not None else [y_base_transform]
        )


    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        x_raw = row[self.x_key]
        y_raw = row[self.y_key]
        x = self.x_transform(x_raw)
        y = self.y_transform(y_raw)
        return x, y

def _em_parquet_dataloader_impl(
    root: str,
    x_key: str,
    y_key: str,
    batch_size: int,
    s_curve_downsample_factor: int,
    num_workers: int,
    x_enable_scale_to_minus1_1: bool = True,
    x_enable_unsqueeze_channel: bool = False,
    y_enable_frequency_crop: bool = False,
    y_original_start_freq: float = 2.0,
    y_original_end_freq: float = 20.0,
    y_original_num_points: int = 1000,
    y_crop_start_freq: Optional[float] = None,
    y_crop_end_freq: Optional[float] = None,
    y_enable_stat_normalize: bool = False,
    y_normalize_mean: float = 0.0,
    y_normalize_max: float = 1.0,
    y_normalize_min: float = 0.0,
    train_filename: str = "train.parquet",
    val_filename: str = "val.parquet",
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    drop_last: bool = True,
    # 可选：与你现有训练框架对齐（如果你后续接 DDP sampler）
    train_sampler=None,
    val_sampler=None,
) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    基于 EMTopologySCurveParquetDataset 构建 train/val 的 DataLoader。

    期望目录结构：
      root/
        train.parquet
        val.parquet

    返回：
      train_loader, val_loader
    """
    if persistent_workers is None:
        # num_workers>0 才能启用 persistent_workers
        persistent_workers = num_workers > 0

    train_path = os.path.join(root, train_filename)
    val_path = os.path.join(root, val_filename)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"未找到训练集文件：{train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"未找到验证集文件：{val_path}")

    x_transform=get_x_transforms(
        enable_scale_to_minus1_1=x_enable_scale_to_minus1_1,
        enable_unsqueeze_channel=x_enable_unsqueeze_channel,
    )
    y_transform=get_y_transforms(
            downsample_factor=s_curve_downsample_factor,
            enable_frequency_crop=y_enable_frequency_crop,
            original_start_freq=y_original_start_freq,
            original_end_freq=y_original_end_freq,
            original_num_points=y_original_num_points,
            crop_start_freq=y_crop_start_freq,
            crop_end_freq=y_crop_end_freq,
            enable_stat_normalize=y_enable_stat_normalize,
            normalize_mean=y_normalize_mean,
            normalize_max=y_normalize_max,
            normalize_min=y_normalize_min,
        )
    
    trainset = EMTopologySCurveParquetDataset(
        parquet_path=train_path,
        x_key=x_key,
        y_key=y_key,
        x_transform=x_transform,
        y_transform=y_transform,
    )
    valset = EMTopologySCurveParquetDataset(
        parquet_path=val_path,
        x_key=x_key,
        y_key=y_key,
        x_transform=x_transform,
        y_transform=y_transform,
    )

    _logger.info(
        f"[EM] Loading Parquet datasets: "
        f"train(len={len(trainset)}) val(len={len(valset)})"
    )

    train_loader = data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )

    val_loader = data.DataLoader(
        dataset=valset,
        batch_size=batch_size,
        shuffle=False,              # 验证集通常不 shuffle
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    return train_loader, val_loader

@DATA.register
def EMParquetDataloader(
    root: str,
    x_key: str,
    y_key: str,
    batch_size: int,
    s_curve_downsample_factor: int,
    num_workers: int,
    x_enable_scale_to_minus1_1: bool = True,
    x_enable_unsqueeze_channel: bool = False,
    y_enable_frequency_crop: bool = False,
    y_original_start_freq: float = 2.0,
    y_original_end_freq: float = 20.0,
    y_original_num_points: int = 1000,
    y_crop_start_freq: Optional[float] = None,
    y_crop_end_freq: Optional[float] = None,
    y_enable_stat_normalize: bool = False,
    y_normalize_mean: float = 0.0,
    y_normalize_max: float = 1.0,
    y_normalize_min: float = 0.0,
    train_filename: str = "train.parquet",
    val_filename: str = "val.parquet",
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    drop_last: bool = True,
    **kwargs):
    kwargs.pop("config", None)
    return _em_parquet_dataloader_impl(
        root=root,
        x_key=x_key,
        y_key=y_key,
        batch_size=batch_size,
        s_curve_downsample_factor=s_curve_downsample_factor,
        num_workers=num_workers,
        x_enable_scale_to_minus1_1=x_enable_scale_to_minus1_1,
        x_enable_unsqueeze_channel=x_enable_unsqueeze_channel,
        y_enable_frequency_crop=y_enable_frequency_crop,
        y_original_start_freq=y_original_start_freq,
        y_original_end_freq=y_original_end_freq,
        y_original_num_points=y_original_num_points,
        y_crop_start_freq=y_crop_start_freq,
        y_crop_end_freq=y_crop_end_freq,
        y_enable_stat_normalize=y_enable_stat_normalize,
        y_normalize_mean=y_normalize_mean,
        y_normalize_max=y_normalize_max,
        y_normalize_min=y_normalize_min,
        train_filename=train_filename,
        val_filename=val_filename,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        **kwargs,
    )
