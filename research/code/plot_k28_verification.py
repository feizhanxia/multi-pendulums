#!/usr/bin/env python3
"""
Generate N=5 K#28 verification chart (4 seeds comparison)
Output: n5_top1_verification_en.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# K#28 verification data
seeds = [0, 42, 123, 456]
selectivities = [7.30, 7.30, 7.30, 7.30]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax.bar([f'Seed {s}' for s in seeds], selectivities, color=colors, edgecolor='black')

ax.set_xlabel('Random Seed', fontsize=12)
ax.set_ylabel('Selectivity (x)', fontsize=12)
ax.set_title('N=5 K#28 Robustness Test: 4 Seeds (Best Omega=1.5)', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 10)

for bar, sel in zip(bars, selectivities):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
            f'{sel:.2f}x', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('n5_top1_verification_en.png', dpi=150)
print('Saved: n5_top1_verification_en.png')
