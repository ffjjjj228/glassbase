import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

N_FREQ = 60
PER_FREQ = 4
# S11 is the first element per frequency (index 0)
S11_IDX = [i for i in range(0, N_FREQ * PER_FREQ, PER_FREQ)]

INPUT = "data/S2P.parquet"
TRAIN_OUT = "data/train_s2p.parquet"
VAL_OUT = "data/val_s2p.parquet"
TEST_OUT = "data/test_s2p.parquet"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

df = pq.read_table(INPUT).to_pandas()
n = len(df)
print(f"Total samples: {n}")

df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)
train_df = df.iloc[:n_train].reset_index(drop=True)
val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

def compute_s11_mag(s_real_raw, s_imag_raw):
    s11_real = np.asarray(s_real_raw, dtype=np.float32)[S11_IDX]
    s11_imag = np.asarray(s_imag_raw, dtype=np.float32)[S11_IDX]
    return np.sqrt(s11_real ** 2 + s11_imag ** 2).tolist()

def build_table(df):
    topo_col = pa.array(
        df["topology_top_flat"].tolist(),
        type=pa.list_(pa.int8(), 484),
    )
    s11_col = pa.array(
        [compute_s11_mag(r, i) for r, i in zip(df["s_real"], df["s_imag"])],
        type=pa.list_(pa.float32(), 60),
    )
    table = pa.Table.from_arrays(
        [topo_col, s11_col],
        names=["topology_top_flat", "s11_mag"],
    )
    meta = {
        "topology_top_flat_shape": "[22,22]",
        "topology_top_flat_dtype": "int8",
        "s11_mag_shape": "[60,]",
        "s11_mag_dtype": "float32",
    }
    table = table.replace_schema_metadata(
        {k.encode(): str(v).encode() for k, v in meta.items()}
    )
    return table

pq.write_table(build_table(train_df), TRAIN_OUT, compression="zstd")
pq.write_table(build_table(val_df), VAL_OUT, compression="zstd")
pq.write_table(build_table(test_df), TEST_OUT, compression="zstd")
print(f"Saved {TRAIN_OUT}")
print(f"Saved {VAL_OUT}")
print(f"Saved {TEST_OUT}")
