# -*- coding: utf-8 -*-
"""Generate Fig. 2: Impact of object eligibility criteria and aggregation methods."""
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Styling for IEEE Letters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.labelsize'] = 9.5
plt.rcParams['xtick.labelsize'] = 8.5
plt.rcParams['ytick.labelsize'] = 8.5
plt.rcParams['legend.fontsize'] = 8.0
plt.rcParams['figure.titlesize'] = 10.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.45), dpi=300, gridspec_kw={'width_ratios': [1.18, 1.0]})

# Panel A: 3x3 Heatmap Matrix
# Data: rows = tau (0.25, 0.50, 0.75), cols = tolerance (0.5m, 1.0m, 2.0m)
data_matrix = np.array([
    [0.269, 0.191, -0.040],
    [0.220, 0.166,  0.005],
    [-0.096, -0.089, -0.190]
])

cmap = plt.cm.coolwarm
vmax = 0.35
vmin = -0.35

im = ax1.imshow(data_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

tol_labels = ['0.5 m', '1.0 m', '2.0 m']
tau_labels = [r'$\tau = 0.25$ (766)', r'$\tau = 0.50$ (741)', r'$\tau = 0.75$ (692)']

ax1.set_xticks(range(3))
ax1.set_xticklabels(tol_labels)
ax1.set_yticks(range(3))
ax1.set_yticklabels(tau_labels)
ax1.set_xlabel('Matching Physical Tolerance')
ax1.set_ylabel('Object IoU Threshold (Shared Objs)')
ax1.set_title('(a) Siamese - Standard U-Net on 342 Tiles', fontsize=9.2, pad=6)

# Overlay numeric difference in each cell
for i in range(3):
    for j in range(3):
        val = data_matrix[i, j]
        sign = '+' if val > 0 else ''
        text_color = 'white' if abs(val) > 0.20 else 'black'
        ax1.text(j, i, f'{sign}{val:.3f}', ha='center', va='center', color=text_color, fontsize=8.5, fontweight='bold')

cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('BF1 difference (pp)', fontsize=8.5)
cbar.ax.tick_params(labelsize=8)

# Panel B: Aggregation Comparison on Identical 692 Objects (tau = 0.75)
tolerances = np.array([0.5, 1.0, 2.0])
pooled = np.array([-0.096, -0.089, -0.190])
obj_equal = np.array([0.492, 0.076, -0.008])

x = np.arange(len(tolerances))
width = 0.32

rects1 = ax2.bar(x - width/2, pooled, width, label='Pooled (Contour counts)', color='#4a7bb7', edgecolor='none', alpha=0.9)
rects2 = ax2.bar(x + width/2, obj_equal, width, label='Object-Equal Mean', color='#e58338', edgecolor='none', alpha=0.9)

ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_xticks(x)
ax2.set_xticklabels(tol_labels)
ax2.set_xlabel('Matching Physical Tolerance')
ax2.set_ylabel('BF1 difference (pp)')
ax2.set_title('(b) Aggregation Impact on Identical 692 Objects', fontsize=9.2, pad=6)
ax2.legend(loc='upper right', frameon=True, framealpha=0.92)
ax2.set_ylim(-0.24, 0.58)

# Add values above/below bars
for rect in rects1:
    h = rect.get_height()
    offset = -0.05 if h < 0 else 0.015
    sign = '+' if h > 0 else ''
    ax2.text(rect.get_x() + rect.get_width()/2., h + offset, f'{sign}{h:.3f}', ha='center', va='bottom', fontsize=7.2, color='#2c4c75', fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    offset = -0.05 if h < 0 else 0.015
    sign = '+' if h > 0 else ''
    ax2.text(rect.get_x() + rect.get_width()/2., h + offset, f'{sign}{h:.3f}', ha='center', va='bottom', fontsize=7.2, color='#944d15', fontweight='bold')

plt.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, 'fig2_eligibility_aggregation.pdf')
png_path = os.path.join(out_dir, 'fig2_eligibility_aggregation.png')

fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, bbox_inches='tight', dpi=300)
print(f'Successfully exported: {pdf_path}')
