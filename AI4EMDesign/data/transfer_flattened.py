#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将原始 CSV 转成更工程化的 Parquet：
- topology_flat: 固定长度 16 的向量（int8）
- s_curve: 固定长度 1000 的向量（float32）

并将 shape 信息写入 Parquet metadata：
- topology_flat_shape = [4,4]
- topology_flat_dtype = int8
- s_curve_shape = [1000,]
- s_curve_dtype = float32
"""

import argparse
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _infer_id_and_payload(df: pd.DataFrame, payload_len: int) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    ncols = df.shape[1]
    if ncols == payload_len:
        return None, df
    if ncols == payload_len + 1:
        return df.iloc[:, 0], df.iloc[:, 1:]
    raise ValueError(
        f"列数不符合预期：期望 {payload_len} 或 {payload_len + 1} 列，但实际为 {ncols} 列"
    )


def _validate_binary(x: np.ndarray) -> None:
    if not np.isin(x, [0, 1]).all():
        bad = x[~np.isin(x, [0, 1])]
        raise ValueError(f"topology 中包含非 0/1 值，例如：{bad[:10]}")


def _build_metadata(extra: Optional[dict] = None) -> dict[bytes, bytes]:
    meta = {
        "topology_flat_shape": "[4,4]",
        "topology_flat_dtype": "int8",
        "s_curve_shape": "[1000,]",
        "s_curve_dtype": "float32",
    }
    if extra:
        meta.update(extra)
    return {k.encode("utf-8"): str(v).encode("utf-8") for k, v in meta.items()}


def build_parquet(
    topo_csv: str,
    s_csv: str,
    out_parquet: str,
    compression: str = "zstd",
) -> None:
    topo_df = pd.read_csv(topo_csv)
    s_df = pd.read_csv(s_csv)

    topo_id, topo_payload = _infer_id_and_payload(topo_df, payload_len=16)
    s_id, s_payload = _infer_id_and_payload(s_df, payload_len=1000)

    if len(topo_payload) != len(s_payload):
        raise ValueError(f"样本数量不一致：topology={len(topo_payload)}，s_curve={len(s_payload)}")

    use_id = None
    if topo_id is not None and s_id is not None:
        if not topo_id.reset_index(drop=True).equals(s_id.reset_index(drop=True)):
            raise ValueError("topology 与 s_curve 的 id 列不一致")
        use_id = topo_id.reset_index(drop=True)
    elif topo_id is not None:
        use_id = topo_id.reset_index(drop=True)
    elif s_id is not None:
        use_id = s_id.reset_index(drop=True)

    topo_arr = topo_payload.to_numpy(dtype=np.int8, copy=True)
    s_arr = s_payload.to_numpy(dtype=np.float32, copy=True)

    if topo_arr.shape[1] != 16:
        raise ValueError(f"topology 列数必须为 16，实际为 {topo_arr.shape[1]}")
    if s_arr.shape[1] != 1000:
        raise ValueError(f"s_curve 列数必须为 1000，实际为 {s_arr.shape[1]}")

    _validate_binary(topo_arr)

    if np.nanmin(s_arr) < 0.0 or np.nanmax(s_arr) > 1.0:
        print(
            f"[WARN] s_curve 超出 [0,1]: min={np.nanmin(s_arr):.6f}, max={np.nanmax(s_arr):.6f}",
            file=sys.stderr,
        )

    arrays = []
    names = []

    if use_id is not None:
        arrays.append(pa.array(use_id.tolist()))
        names.append("id")

    arrays.append(pa.array(topo_arr.tolist(), type=pa.list_(pa.int8(), 16)))
    names.append("topology_flat")
    arrays.append(pa.array(s_arr.tolist(), type=pa.list_(pa.float32(), 1000)))
    names.append("s_curve")

    table = pa.Table.from_arrays(arrays, names=names)
    table = table.replace_schema_metadata(_build_metadata())

    pq.write_table(
        table,
        out_parquet,
        compression=compression,
        use_dictionary=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 topology_flat(16) + metadata 的 Parquet")
    parser.add_argument("--topo_csv", required=True, help="拓扑 CSV 路径（16 列或 id+16 列）")
    parser.add_argument("--s_csv", required=True, help="S 参数 CSV 路径（1000 列或 id+1000 列）")
    parser.add_argument("--out", required=True, help="输出 Parquet 路径")
    parser.add_argument("--compression", default="zstd", help="压缩方式（zstd/snappy/gzip）")
    args = parser.parse_args()

    build_parquet(
        topo_csv=args.topo_csv,
        s_csv=args.s_csv,
        out_parquet=args.out,
        compression=args.compression,
    )
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
