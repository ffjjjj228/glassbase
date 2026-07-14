import pandas as pd

df = pd.read_parquet("data/topology_s4p_toponly.parquet")

assert len(df) >= 100, f"数据量不足，当前只有 {len(df)} 条"

df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

split = int(len(df) * 0.8)
train_df = df.iloc[:split]
val_df   = df.iloc[split:]

train_df.to_parquet("data/train_s4p.parquet", index=False)
val_df.to_parquet("data/val_s4p.parquet", index=False)

print(f"Saved train_s4p.parquet (len={len(train_df)})")
print(f"Saved val_s4p.parquet   (len={len(val_df)})")
