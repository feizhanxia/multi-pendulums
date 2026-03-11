"""
主脚本：集中配置参数，可选择运行单次仿真、参数扫描和可视化。
按需修改“用户区”参数，并取消/保留对应的 run_xxx 标志即可。
"""

from __future__ import annotations

from pathlib import Path

from analyze import main as analyze_main
from scan import main as scan_main
from simulate import main as simulate_main

# ========== 用户区：统一参数 ==========
# 单次仿真参数（simulate.py）
SIM_ARGS = [
    "--N", "5",
    "--gamma", "0.08",
    "--F", "0.0025",
    # "--K", "0.95",
    # "--Omega", "0.55",
    "--K", "0.9999",
    "--Omega", "0.51",
    "--w0", "1.0",
    "--detune_eps", "0.12",   # 小频率梯度，轻度打破简并
    "--detune_target", "2",
    "--t_total", "2500",
    "--dt", "0.025",
    "--discard_ratio", "0.6",
    "--seed", "0",
    "--drive_index", "0",
    "--summary_path", "data/summary/single_case.csv",
    "--plot_path", "figures/single_case.png",
    # 如需保存原始时间序列，取消注释
    # "--save_raw",
    # "--raw_path", "data/raw/single_case.npz",
]

# 扫描参数（scan.py）
SCAN_ARGS = [
    "--N", "5",
    "--gamma", "0.08",
    "--F", "0.0025",             # 再减小驱动幅值
    "--w0", "1.0",
    "--detune_eps", "0.12",      # 更强定向频率偏置
    "--detune_target", "2",      # 只对节点2做偏置（非驱动）
    "--t_total", "2500",         # 更长积分时间
    "--dt", "0.025",             # 更细采样
    "--discard_ratio", "0.6",    # 丢弃更多初段以保证稳态
    "--seed", "0",
    "--drive_index", "0",
    # 频率/耦合网格（极小窗口）
    "--omega_range", "0.3", "1.16", "0.01",  # start stop step
    "--k_range", "0.01", "1.0", "0.01",
    "--workers", "8",
    "--summary_path", "data/summary/grid_n5_detune.csv",
    # 如需保存原始时间序列，请取消下一行注释（体积较大
    # "--save_raw",
    # "--raw_dir", "data/raw/grid",
]

# 可视化参数（analyze.py）
ANALYZE_ARGS = [
    "--summary_path", "data/summary/grid_n5_detune.csv",
    "--metric", "selectivity_fft_nd",  # 使用主频选择性（排除驱动）
    "--top", "3",
    "--out_dir", "figures",
]

# 运行开关：将对应标志设为 True/False 以选择执行的步骤。
RUN_SINGLE = True      # 单次仿真
RUN_SCAN = False       # 参数扫描
RUN_ANALYZE = False    # 生成可视化（基于 summary）
# =====================================


def main() -> None:
    # 单次仿真
    if RUN_SINGLE:
        print("=== Running single simulation ===")
        simulate_main(SIM_ARGS)

    # 参数扫描
    if RUN_SCAN:
        print("=== Running grid scan ===")
        scan_main(SCAN_ARGS)

    # 可视化
    if RUN_ANALYZE:
        print("=== Running analysis/plots ===")
        analyze_main(ANALYZE_ARGS)


if __name__ == "__main__":
    main()
