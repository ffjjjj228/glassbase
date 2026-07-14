import argparse
import numpy as np
import pandas as pd
from prettytable import PrettyTable


def analyze_sparam_distribution(
    data_path: str,
    s_column: str = "s_curve",
    bins: int = 10,
):
    """
    分析 S 参数分布情况，并用 PrettyTable 打印

    参数
    ----
    data_path : str
        parquet 或 csv 文件路径
    s_column : str
        S 参数列名（默认 s_curve）
    bins : int
        分桶数量（默认 10，对应 0.0-1.0 每 0.1 一个区间）
    """

    # 读取数据
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("仅支持 parquet 或 csv 文件")

    if s_column not in df.columns:
        raise ValueError(f"列 {s_column} 不存在，可选列为 {df.columns}")

    num_samples = len(df)

    # 展平所有 S 参数
    # shape: (num_samples * 1000,)
    all_s = np.concatenate(df[s_column].values)

    # 定义区间
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    counts, _ = np.histogram(all_s, bins=bin_edges)
    total_points = all_s.size

    # 每个样本的统计
    per_sample_counts = []
    for s in df[s_column].values:
        hist, _ = np.histogram(s, bins=bin_edges)
        per_sample_counts.append(hist)

    per_sample_counts = np.stack(per_sample_counts)
    per_sample_mean = per_sample_counts.mean(axis=0)

    # PrettyTable
    table = PrettyTable()
    table.field_names = [
        "S 参数区间",
        "总频点数",
        "占比 (%)",
        "每样本平均频点数",
    ]

    for i in range(bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        count = int(counts[i])
        ratio = count / total_points * 100.0
        avg_per_sample = per_sample_mean[i]

        table.add_row([
            f"[{low:.1f}, {high:.1f})",
            count,
            f"{ratio:.2f}",
            f"{avg_per_sample:.1f}",
        ])

    return table, num_samples, total_points


def main():
    parser = argparse.ArgumentParser("Analyze S-parameter distribution (PrettyTable)")
    parser.add_argument("--data", required=True, help="parquet / csv 文件路径")
    parser.add_argument("--s_column", default="s_curve", help="S 参数列名")
    parser.add_argument("--bins", type=int, default=10, help="分桶数量（默认10）")

    args = parser.parse_args()

    table, num_samples, total_points = analyze_sparam_distribution(
        data_path=args.data,
        s_column=args.s_column,
        bins=args.bins,
    )

    print("\n===== S 参数分布统计（PrettyTable）=====\n")
    print(f"样本数量        : {num_samples}")
    print(f"总频点数量      : {total_points}")
    print(f"每样本频点数量  : {total_points // num_samples}")
    print()
    print(table)
    print("\n说明：")
    print("- 总频点数：全数据集中落在该区间的频点数量")
    print("- 占比 (%)：该区间频点占所有频点的比例")
    print("- 每样本平均频点数：单个样本在该区间内的平均频点数量")


if __name__ == "__main__":
    main()
