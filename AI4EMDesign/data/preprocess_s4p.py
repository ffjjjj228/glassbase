import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

N_FREQ = 60
PER_FREQ = 16
# Upper triangle of 4x4 S-matrix (reciprocity): 10 unique per frequency
UNIQUE_IDX = [0, 1, 2, 3, 5, 6, 7, 10, 11, 15]

IDX_LIST = [i + f * PER_FREQ for f in range(N_FREQ) for i in UNIQUE_IDX]

INPUT = "data/topology_s4p_toponly.parquet"
TRAIN_OUT = "data/train_s4p_proc.parquet"
VAL_OUT = "data/val_s4p_proc.parquet"
TEST_OUT = "data/test_s4p_proc.parquet"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

df_list = pq.read_table(INPUT).to_pandas()
n = len(df_list)
print(f"Total samples: {n}")

# Shuffle
df_list = df_list.sample(frac=1.0, random_state=42).reset_index(drop=True)

n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)
train_df = df_list.iloc[:n_train].reset_index(drop=True)
val_df = df_list.iloc[n_train:n_train + n_val].reset_index(drop=True)
test_df = df_list.iloc[n_train + n_val:].reset_index(drop=True)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

def process_row(s_real_raw, s_imag_raw):
    arr_real = np.asarray(s_real_raw, dtype=np.float32)[IDX_LIST]
    arr_imag = np.asarray(s_imag_raw, dtype=np.float32)[IDX_LIST]
    return np.concatenate([arr_real, arr_imag]).tolist()

def validate_border(topo_flat):
    topo = np.asarray(topo_flat, dtype=np.int8).reshape(22, 22)
    top_edge = topo[0].sum()
    bottom_edge = topo[21].sum()
    left_edge = topo[1:21, 0].sum()
    right_edge = topo[1:21, 21].sum()
    if not (top_edge == 1 and bottom_edge == 1 and left_edge == 1 and right_edge == 1):
        raise ValueError(f"Border constraint violated: top={top_edge}, bottom={bottom_edge}, left={left_edge}, right={right_edge}")

def build_table(df):
    for _, row in df.iterrows():
        validate_border(row["topology_top_flat"])
    print(f"Border constraint: all {len(df)} samples valid")

    topo_col = pa.array(
        df["topology_top_flat"].tolist(),
        type=pa.list_(pa.int8(), 484),
    )
    s_curve_col = pa.array(
        [process_row(r, i) for r, i in zip(df["s_real"], df["s_imag"])],
        type=pa.list_(pa.float32(), 1200),
    )
    table = pa.Table.from_arrays(
        [topo_col, s_curve_col],
        names=["topology_top_flat", "s_curve"],
    )
    meta = {
        "topology_top_flat_shape": "[22,22]",
        "topology_top_flat_dtype": "int8",
        "s_curve_shape": "[1200,]",
        "s_curve_dtype": "float32",
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
