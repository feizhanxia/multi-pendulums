# N=5 Kij Strict Search (Clean Repro)

## 目标
- 在 N=5 耦合摆网络中搜索非对称 `K_ij`，最大化非驱动节点主频响应选择性。
- 用严格四阶段流程筛掉“高分但不稳”的候选。

## 完整计算实验流程
1. 粗扫描（Coarse Search）
- 随机采样 `n_samples=35` 个 `K_ij`（范围 `[-1, 1]`，对角线置零）。
- 对每个 `K_ij` 在 `Omega=0.5:0.1:1.5` 上扫描。
- 记录每个候选的最佳 `Omega` 和最佳选择性（`selectivity_fft_nd`）。
- 选 Top-5 进入下一步。

2. 鲁棒性初筛（Robustness Gate）
- 对 Top-5 在其粗扫最佳 `Omega` 下，使用 `seed=0,42` 复验。
- 若两 seed 的最小选择性 `>2.0`，判定通过。

3. 细化扫描（Refined Search）
- 对通过初筛的候选在 `Omega=0.5:0.02:1.5` 细扫。
- 取细扫最佳 `Omega`，再用 `seed=0,42` 复验。
- 最小选择性 `>2.0` 才进入最终验证。

4. 最终验证（Final Verification）
- 对最终候选在细扫最佳 `Omega` 下使用 `seed=0,42,123,456` 验证。
- 输出 `min/avg selectivity`，并标记 `stable`。

## 指标定义
- 每个节点响应：稳态尾段在驱动频率处的 FFT 幅值 `amp_fft[i]`。
- 非驱动选择性：`max(non_drive_amp_fft) / second_max(non_drive_amp_fft)`。

## 目录结构
- `src/model_kij.py`：动力学模型。
- `src/simulate_kij.py`：单次仿真与指标计算。
- `scripts/run_strict_search.py`：四阶段严格搜索主脚本。
- `scripts/plot_strict_results.py`：生成 `n5_top1_verification_en.png`。
- `data/n5_strict_results.json`：结构化结果。
- `figures/n5_top1_verification_en.png`：双子图（频响+鲁棒柱状图）。

## 运行
```bash
cd n5_kij_research_clean
./run_all.sh
```

## 依赖
- Python 3.10+
- `numpy`, `scipy`, `matplotlib`
