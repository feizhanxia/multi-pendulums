"""
主脚本：集中配置 K_ij 新模块参数，可选择运行单次仿真、Omega 扫描与分析。
按需修改“用户区”参数，并取消/保留对应的 RUN_xxx 标志即可。
"""

from __future__ import annotations

from analyze_kij import main as analyze_main
from plot_kij import main as plot_main
from scan_kij import main as scan_main
from search_kij import main as search_main
from simulate_kij import main as simulate_main

# ========== 用户区：统一参数 ==========
# 单次仿真参数（simulate_kij.py）
SIM_ARGS = [
    "--N",
    "3",
    "--gamma",
    "0.08",
    "--F",
    "0.1",
    "--Omega",
    "1.1",
    "--w0",
    "1.0",
    "--K",
    "0",
    "-0.8844710092276815",
    "0.3174935666956553",
    "-0.5590284438039554",
    "0",
    "0.8135494042112927",
    "0.9758396075210554",
    "-0.9062683442871464",
    "0",
    "--t_total",
    "800",
    "--dt",
    "0.1",
    "--discard_ratio",
    "0.5",
    "--summary_path",
    "data/summary/single_kij.csv",
    "--plot_path",
    "figures/kij_run/single_kij.png",
]

# 扫描参数（scan_kij.py）
SCAN_ARGS = [
    "--N",
    "3",
    "--gamma",
    "0.08",
    "--F",
    "0.1",
    "--w0",
    "1.0",
    "--K",
    "0",
    "-0.8844710092276815",
    "0.3174935666956553",
    "-0.5590284438039554",
    "0",
    "0.8135494042112927",
    "0.9758396075210554",
    "-0.9062683442871464",
    "0",
    "--omega_range",
    "0.6",
    "1.4",
    "0.01",
    "--dt",
    "0.1",
    "--summary_path",
    "data/summary/scan_kij.csv",
]

# 分析参数（analyze_kij.py）
ANALYZE_ARGS = [
    "--summary_path",
    "data/summary/scan_kij.csv",
    "--node_a",
    "1",
    "--node_b",
    "2",
    "--window_size",
    "15",
    "--out_dir",
    "figures/kij_run",
]

# 矩阵搜索参数（search_kij.py）
SEARCH_ARGS = [
    "--N",
    "3",
    "--gamma",
    "0.08",
    "--F",
    "0.1",
    "--w0",
    "1.0",
    "--method",
    "bo",
    "--k_range",
    "-1.0",
    "1.0",
    "--n_init",
    "8",
    "--n_iter",
    "12",
    "--n_candidates",
    "200",
    "--bo_kappa",
    "2.0",
    "--omega_range",
    "0.6",
    "1.4",
    "0.03",
    "--window_size",
    "5",
    "--node_a",
    "1",
    "--node_b",
    "2",
    "--t_total",
    "800",
    "--dt",
    "0.1",
    "--discard_ratio",
    "0.5",
    "--top_k",
    "5",
    "--out_root",
    "data/summary/search_runs",
]

# K_ij 可视化参数（plot_kij.py）
PLOT_ARGS = [
    "--N",
    "3",
    "--K",
    "0",
    "-0.8844710092276815",
    "0.3174935666956553",
    "-0.5590284438039554",
    "0",
    "0.8135494042112927",
    "0.9758396075210554",
    "-0.9062683442871464",
    "0",
    "--out_path",
    "figures/kij_run/kij_heatmap.png",
    "--title",
    "K_ij heatmap (top result 20260114_150725)",
]

# 运行开关：将对应标志设为 True/False 以选择执行的步骤。
# RUN_SINGLE = False
RUN_SCAN = False
RUN_ANALYZE = False
# RUN_SEARCH = True
RUN_PLOT = False

RUN_SINGLE = True
# RUN_SCAN = True
# RUN_ANALYZE = True
RUN_SEARCH = False
# RUN_PLOT = True
# =====================================


def main() -> None:
    if RUN_SINGLE:
        print("=== Running single K_ij simulation ===")
        simulate_main(SIM_ARGS)

    if RUN_SCAN:
        print("=== Running K_ij omega scan ===")
        scan_main(SCAN_ARGS)

    if RUN_ANALYZE:
        print("=== Running K_ij analysis ===")
        analyze_main(ANALYZE_ARGS)

    if RUN_SEARCH:
        print("=== Running K_ij matrix search ===")
        search_main(SEARCH_ARGS)

    if RUN_PLOT:
        print("=== Plotting K_ij heatmap ===")
        plot_main(PLOT_ARGS)


if __name__ == "__main__":
    main()
