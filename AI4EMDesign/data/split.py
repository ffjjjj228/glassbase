import pandas as pd

# 读取完整数据
df = pd.read_parquet("data/dataset_flat.parquet")

# 基本检查
assert len(df) >= 2200, f"数据量不足，当前只有 {len(df)} 条"

# 打乱（固定随机种子，保证可复现）
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 切分
train_df = df.iloc[:2000]
val_df   = df.iloc[2000:2200]

# 保存
train_df.to_parquet("data/train_flat.parquet", index=False)
val_df.to_parquet("data/val_flat.parquet", index=False)

print(f"Saved train_flat.parquet (len={len(train_df)})")
print(f"Saved val_flat.parquet   (len={len(val_df)})")
