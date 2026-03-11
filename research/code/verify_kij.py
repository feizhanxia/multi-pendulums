#!/usr/bin/env python3
"""验证 K_ij 优化结果的可靠性 - 无pandas版本"""

import json
import csv
import numpy as np
from pathlib import Path

from model_kij import ParamsKij, pendulum_ode_kij
from scipy.integrate import solve_ivp

# 最优 K_ij
K_opt = [
    [0, -0.447, 0.421],
    [-0.612, 0, -0.899],
    [0.615, -0.679, 0]
]

OUTPUT_DIR = Path("/home/yono/.openclaw/workspace/kij_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_single_sim(Omega, seed, t_total=800, dt=0.1, discard_ratio=0.5):
    """运行单次仿真"""
    params = ParamsKij(
        N=3,
        gamma=0.08,
        F=0.1,
        Omega=Omega,
        w0=1.0,
        K=K_opt,
        discard_ratio=discard_ratio,
        t_total=t_total,
        dt=dt,
        seed=seed,
        drive_index=0,
    )
    
    t_eval = np.arange(0.0, params.t_total, params.dt)
    if t_eval[-1] < params.t_total:
        t_eval = np.append(t_eval, params.t_total)
    
    # 初始条件
    rng = np.random.default_rng(params.seed)
    noise_scale = 1e-3
    theta0 = noise_scale * rng.standard_normal(params.N)
    omega0 = noise_scale * rng.standard_normal(params.N)
    y0 = np.concatenate([theta0, omega0])
    
    sol = solve_ivp(
        lambda t, y: pendulum_ode_kij(t, y, params),
        (0.0, params.t_total),
        y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )
    
    theta = sol.y[:params.N].T
    discard_idx = int(len(t_eval) * discard_ratio)
    theta_tail = theta[discard_idx:]
    
    # 计算振幅 (RMS)
    amp = np.sqrt(np.mean(theta_tail**2, axis=0))
    
    # FFT分析
    phase = np.exp(-1j * params.Omega * t_eval[discard_idx:])
    amp_fft = np.abs(np.mean(theta_tail * phase[:, None], axis=0))
    
    return {
        "Omega": Omega,
        "seed": seed,
        "amp_node0": float(amp[0]),
        "amp_node1": float(amp[1]),
        "amp_node2": float(amp[2]),
        "amp_fft_node0": float(amp_fft[0]),
        "amp_fft_node1": float(amp_fft[1]),
        "amp_fft_node2": float(amp_fft[2]),
        "selectivity": float(amp_fft[1] / (amp_fft[2] + 1e-12)),
    }

def save_csv(filename, data, fieldnames):
    """保存为CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def main():
    print("=" * 60)
    print("K_ij 优化结果验证")
    print("=" * 60)
    
    # ========== 1. 细化频率扫描 ==========
    print("\n[1] 细化频率扫描: Ω ∈ [0.9, 1.3], 步长 0.02")
    print("-" * 40)
    
    omega_values = np.arange(0.9, 1.32, 0.02)
    frequency_results = []
    
    for omega in omega_values:
        result = run_single_sim(omega, seed=0)
        frequency_results.append(result)
        print(f"  Ω={omega:.2f}: amp1={result['amp_node1']:.4f}, amp2={result['amp_node2']:.4f}, sel={result['selectivity']:.1f}x")
    
    # 保存CSV
    csv_path = OUTPUT_DIR / "frequency_scan.csv"
    save_csv(csv_path, frequency_results, frequency_results[0].keys())
    print(f"\n  → 保存到: {csv_path}")
    
    # 找到最佳频率
    best_selectivity = 0
    best_omega = 0
    for r in frequency_results:
        if r['selectivity'] > best_selectivity:
            best_selectivity = r['selectivity']
            best_omega = r['Omega']
    
    print(f"\n  ★ 最佳频率: Ω = {best_omega:.2f}")
    print(f"  ★ 最高选择性: {best_selectivity:.1f}x")
    
    # ========== 2. 鲁棒性测试 ==========
    print("\n[2] 鲁棒性测试: seed = 0, 1, 2")
    print("-" * 40)
    
    # 使用最佳频率点进行测试
    test_omega = best_omega
    robustness_results = []
    
    for seed in [0, 1, 2]:
        result = run_single_sim(test_omega, seed=seed)
        robustness_results.append(result)
        print(f"  seed={seed}: amp1={result['amp_node1']:.4f}, amp2={result['amp_node2']:.4f}, sel={result['selectivity']:.1f}x")
    
    csv_path_robust = OUTPUT_DIR / "robustness_test.csv"
    save_csv(csv_path_robust, robustness_results, robustness_results[0].keys())
    print(f"\n  → 保存到: {csv_path_robust}")
    
    # 检查一致性
    selectivities = [r['selectivity'] for r in robustness_results]
    mean_sel = np.mean(selectivities)
    std_sel = np.std(selectivities)
    print(f"\n  选择性统计: mean={mean_sel:.2f}, std={std_sel:.2f}")
    print(f"  差异系数: {std_sel/mean_sel*100:.1f}%")
    
    # ========== 3. 稳态验证 ==========
    print("\n[3] 稳态验证: t_total=1500, discard_ratio=0.6")
    print("-" * 40)
    
    # 使用最佳频率
    steady_result = run_single_sim(
        Omega=test_omega, 
        seed=0, 
        t_total=1500, 
        dt=0.1, 
        discard_ratio=0.6
    )
    
    # 再次运行更长的仿真做对比
    steady_result2 = run_single_sim(
        Omega=test_omega, 
        seed=0, 
        t_total=2000, 
        dt=0.1, 
        discard_ratio=0.6
    )
    
    print(f"  t_total=1500: amp1={steady_result['amp_node1']:.4f}, amp2={steady_result['amp_node2']:.4f}")
    print(f"  t_total=2000: amp1={steady_result2['amp_node1']:.4f}, amp2={steady_result2['amp_node2']:.4f}")
    
    # 计算相对差异
    amp1_diff = abs(steady_result['amp_node1'] - steady_result2['amp_node1']) / steady_result2['amp_node1'] * 100
    amp2_diff = abs(steady_result['amp_node2'] - steady_result2['amp_node2']) / steady_result2['amp_node2'] * 100
    
    print(f"\n  振幅差异: 节点1={amp1_diff:.2f}%, 节点2={amp2_diff:.2f}%")
    
    if amp1_diff < 5 and amp2_diff < 10:
        steady_state = "✓ 稳态收敛"
    else:
        steady_state = "⚠ 可能未完全收敛"
    
    print(f"  结论: {steady_state}")
    
    csv_path_steady = OUTPUT_DIR / "steady_state_test.csv"
    save_csv(csv_path_steady, [steady_result, steady_result2], steady_result.keys())
    print(f"\n  → 保存到: {csv_path_steady}")
    
    # ========== 总结 ==========
    print("\n" + "=" * 60)
    print("验证结果总结")
    print("=" * 60)
    print(f"  最佳频率点: Ω = {best_omega:.2f}")
    print(f"  最高选择性: {best_selectivity:.1f}x")
    print(f"  鲁棒性 (std/mean): {std_sel/mean_sel*100:.1f}%")
    print(f"  稳态验证: {steady_state}")
    print("=" * 60)

if __name__ == "__main__":
    main()
