from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    parquet_path = data_dir / "dataset_flat.parquet"
    if not parquet_path.exists():
        parquet_path = data_dir / "train.parquet"

    df = pd.read_parquet(parquet_path)

    if df.empty:
        print(f"empty parquet file: {parquet_path}")
        return

    parquet_file = pq.ParquetFile(parquet_path)
    raw_meta = parquet_file.schema_arrow.metadata or {}
    meta = {k.decode("utf-8"): v.decode("utf-8") for k, v in raw_meta.items()}

    row = df.iloc[0]
    print(f"file: {parquet_path}")
    print(f"num_rows: {len(df)}")
    print(f"keys: {list(df.columns)}")
    print("metadata:")
    print(f"  topology_flat_shape={meta['topology_flat_shape']}")
    print(f"  topology_flat_dtype={meta['topology_flat_dtype']}")
    print(f"  s_curve_shape={meta['s_curve_shape']}")
    print(f"  s_curve_dtype={meta['s_curve_dtype']}")
    print("-" * 60)

    for key in df.columns:
        value = row[key]
        shape = tuple(eval(meta[f"{key}_shape"]))
        dtype = np.dtype(eval(f"np.{meta[f'{key}_dtype']}"))
        arr = np.asarray(value, dtype=dtype).reshape(shape)
        print(
            f"{key}: raw_type={type(value).__name__}, "
            f"reconstructed_ndarray_dtype={arr.dtype}, reconstructed_ndarray_shape={arr.shape}"
        )
        print(f"{key} raw value:\n{value}")
        print(f"{key} reconstructed value:\n{arr}")
        print()

    # flatten 模式下展示 topology_flat 还原后的 4x4
    if "topology_flat" in df.columns:
        topo_flat = np.asarray(
            row["topology_flat"],
            dtype=np.dtype(eval(f"np.{meta['topology_flat_dtype']}")),
        ).reshape(tuple(eval(meta["topology_flat_shape"])))
        topo_shape = tuple(eval(meta["topology_flat_shape"]))
        topo_matrix = topo_flat.reshape(topo_shape)
        print("topology_flat (reshaped):")
        print(topo_matrix)
        print(f"topology_matrix_shape={topo_matrix.shape}")


if __name__ == "__main__":
    main()
