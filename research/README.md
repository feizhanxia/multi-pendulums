# N=5 耦合摆网络能量选择性传递研究

## 概述

本项目研究 N=5 耦合摆网络中，通过调整非对称耦合矩阵 K_ij 实现能量定向传递的可能性。

## 核心结果

| 参数 | 值 |
|------|-----|
| **最优 K 矩阵** | K#28 |
| **驱动频率 Ω** | 1.5 |
| **选择性** | **7.30x** |
| **鲁棒性** | 4 seed 验证全部通过 |

## K#28 耦合矩阵

```
K = [
  [ 0.000,  0.548, -0.813,  0.941, -0.035],
  [-0.258,  0.000, -0.331,  0.103,  0.063],
  [ 0.631, -0.486,  0.000, -0.724, -0.355],
  [ 0.144, -0.088, -0.084,  0.000,  0.082],
  [-0.199,  0.602,  0.229,  0.511,  0.000]
]
```

---

## 代码文件说明

### 核心模型代码（位于项目根目录）

| 文件 | 功能 |
|------|------|
| `model_kij.py` | 定义 N=3/N=5 耦合摆动力学模型，包含 `ParamsKij` 参数类和 `pendulum_ode_kij` 运动方程 |
| `simulate_kij.py` | 单次仿真：积分 ODE，计算振幅、主频响应、选择性指标 |
| `search_kij.py` | 随机搜索 + 贝叶斯优化：搜索最优 K_ij 矩阵 |
| `scan_kij.py` | 网格扫描：遍历 Ω-K 参数空间 |
| `analyze_kij.py` | 可视化：热力图、柱状图、振幅曲线 |
| `verify_kij.py` | 验证脚本：测试不同 seed 的鲁棒性 |

### 研究专用代码（位于 research/code/）

| 文件 | 功能 |
|------|------|
| `verify_kij.py` | 复制自根目录，用于验证 K#28 的稳定性 |
| `search_kij.py` | 复制自根目录，用于理解搜索算法 |

---

## 研究过程与计算流程

### Step 1: 粗扫描 (Coarse Scan)
- **代码**: `search_kij.py`
- **操作**: 随机生成 35 组 K_ij，遍历 Ω ∈ [0.5, 1.5]，步长 0.1
- **输出**: `n5_strict_results.json` 中的 `coarse_results`
- **结果**: 选出 Top-5 候选

### Step 2: 鲁棒性验证 (Robustness Test)
- **代码**: `verify_kij.py`
- **操作**: 对每个 Top-5 候选，用 seed=0 和 seed=42 运行仿真
- **筛选条件**: 两个 seed 的选择性都必须 > 2x
- **输出**: `n5_strict_results.json` 中的 `robustness_test`
- **结果**: K#21 和 K#28 通过

### Step 3: 细化扫描 (Refined Scan)
- **代码**: `verify_kij.py`
- **操作**: 对鲁棒通过的候选，Ω 步长细化到 0.02
- **输出**: `n5_strict_results.json` 中的 `refined_results`
- **发现**: K#21 细化后崩溃（52x → 1.5x），K#28 稳定

### Step 4: 最终验证 (Final Verification)
- **代码**: `verify_kij.py`
- **操作**: 用 4 个 seed (0, 42, 123, 456) 验证 K#28
- **输出**: `n5_strict_results.json` 中的 `final_verification`
- **结果**: 全部 7.30x，确认稳定

---

## 数据与图表来源

### research/data/

| 文件 | 由什么生成 | 内容 |
|------|------------|------|
| `n5_strict_results.json` | `verify_kij.py` | 完整搜索结果：粗扫描、鲁棒验证、细化扫描、最终验证 |
| `n5_top1_verification.json` | 早期验证脚本 | K#3 的详细验证数据（已被否定）|

### research/figures/

| 文件 | 由什么生成 | 内容 |
|------|------------|------|
| `amplitude_vs_omega_en.png` | 研究过程生成 | N=3 振幅-频率曲线（选择性 30.5x）|
| `n5_scan_figures_en.png` | `analyze_kij.py` | N=5 粗扫描 Top-5 柱状图 |
| `n5_top1_verification_en.png` | 研究过程生成 | N=5 K#28 鲁棒性验证（4 seed 对比）|

---

## 运行顺序

```
1. model_kij.py     # 定义动力学模型（被其他文件 import）
2. simulate_kij.py # 定义仿真函数（被其他文件 import）
3. search_kij.py   # 执行粗搜索（可选）
4. verify_kij.py   # 执行严格筛选流程
   ├── 粗扫描
   ├── 鲁棒验证
   ├── 细化扫描
   └── 最终验证
5. analyze_kij.py  # 可视化结果
```

---

## 复现结果

运行 `verify_kij.py` 即可复现 K#28 的验证结果：

```bash
cd research/code
python verify_kij.py --k_idx 28 --omega 1.5
```

或直接使用 `n5_strict_results.json` 中的参数。

---

## 目录结构

```
multi-pendulums/
├── research/           # N=5 研究
│   ├── README.md     # 本文件
│   ├── report.md     # 完整研究报告
│   ├── code/         # 研究专用代码
│   │   ├── verify_kij.py
│   │   ├── search_kij.py
│   │   └── model_kij.py
│   ├── data/         # 研究数据
│   │   ├── n5_strict_results.json
│   │   └── n5_top1_verification.json
│   └── figures/       # 研究图表
│       ├── amplitude_vs_omega_en.png
│       ├── n5_scan_figures_en.png
│       └── n5_top1_verification_en.png
├── archive/           # 归档
├── model_kij.py      # 核心模型（根目录）
├── simulate_kij.py   # 仿真
├── search_kij.py     # 搜索
└── ...
```

---

## 参考

- 父项目: https://github.com/feizhanxia/multi-pendulums
