import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# Set style similar to run_tests.py
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)    
plt.style.use('ggplot')
mpl.rcParams.update({'font.size': 10})

# Specify the path to the ttf file directly
font_path = './results/cmunrm.ttf' 
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'CMU Serif'

cmap = plt.get_cmap('tab10')
colors = [cmap(value) for value in [*np.linspace(0, 1, 8)]*2]

# Load and merge data
df1 = pd.read_csv('synthetic_missing_results.csv')
if os.path.exists('synthetic_missing_results_0.8.csv'):
    df2 = pd.read_csv('synthetic_missing_results_0.8.csv')
    df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
else:
    df = df1

# Clean up mode for 0.0 rates to match
# Since 0.0 is 'none', we can duplicate it for 'pixel' and 'frame' so they connect nicely in plots
zero_rates = df[df['rate'] == 0.0].copy()
zero_pixel = zero_rates.copy()
zero_pixel['drop_mode'] = 'pixel'
zero_frame = zero_rates.copy()
zero_frame['drop_mode'] = 'frame'
df = pd.concat([df[df['rate'] > 0], zero_pixel, zero_frame]).reset_index(drop=True)

# -------------------------------------------------------------
# Plot 1, 2 & 3: Combined Three Column Plot
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

color_map = {
    'pixel': colors[0], # Red-ish from tab10
    'frame': colors[1]  # Blue-ish
}

label_map = {
    ('pixel', True): 'Random pixels (gapfilled)',
    ('pixel', False): 'Random pixels (zeros)',
    ('frame', True): 'Whole frames (gapfilled)',
    ('frame', False): 'Whole frames (zeros)'
}

for drop_mode in ['pixel', 'frame']:
    for use_gapfill in [True, False]:
        subset = df[(df['drop_mode'] == drop_mode) & (df['use_gapfill'] == use_gapfill)].sort_values('rate')
        if len(subset) == 0: continue
            
        color = color_map[drop_mode]
        ls = '-' if use_gapfill else (0, (2, 2))
        label = label_map[(drop_mode, use_gapfill)]
        
        # Plot BPPPB
        axes[0].plot(subset['rate'] * 100, subset['bpppb'], marker='o', markersize=4,
                 color=color, linestyle=ls, label=label)
                 
        # Plot PSNR
        axes[1].plot(subset['rate'] * 100, subset['psnr'], marker='o', markersize=4,
                 color=color, linestyle=ls, label=label)
                 
        # Plot PSNR vs BPPPB
        axes[2].plot(subset['bpppb'], subset['psnr'], marker='o', markersize=4,
                 color=color, linestyle=ls, label=label)
                 
        # Annotations
        for i, row in subset.iterrows():
            # Don't annotate 0% for every line to avoid clutter, just once or offset
            if row['rate'] == 0.0 and use_gapfill == False: continue
            
            # Annotate PSNR in plot 1
            axes[0].annotate(f"{row['psnr']:.1f} dB", (row['rate'] * 100, row['bpppb']), 
                         xytext=(5, -5), textcoords='offset points', 
                         fontsize=6, color=color)
                         
            # Annotate BPPPB in plot 2
            axes[1].annotate(f"{row['bpppb']:.2f} bpppb", (row['rate'] * 100, row['psnr']), 
                         xytext=(5, -5), textcoords='offset points', 
                         fontsize=6, color=color)
                         
            # Annotate missing rate for the 3rd plot
            rate_str = f"{int(row['rate']*100)}%"
            axes[2].annotate(rate_str, (row['bpppb'], row['psnr']), 
                         xytext=(5, -5), textcoords='offset points', 
                         fontsize=6, color=color)

axes[0].set_xlabel('Missing data (%)')
axes[0].set_ylabel('bpppb')
axes[0].set_ylim(ymin=0)

axes[1].set_xlabel('Missing data (%)')
axes[1].set_ylabel('PSNR (dB)')

axes[2].set_xlabel('bpppb')
axes[2].set_ylabel('PSNR (dB)')

import matplotlib.ticker as mtick
axes[2].set_xscale('log')
axes[2].set_xlim(xmin=0.1, xmax=10.) 
axes[2].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

# Put legend on the bottom right of the rightmost plot (or common)
handles, labels = axes[2].get_legend_handles_labels()
axes[2].legend(handles, labels, loc='lower right', frameon=True, framealpha=0.8, fontsize=8)

plt.tight_layout()
plt.savefig('results/synthetic_missing_composite.pdf', dpi=200, bbox_inches='tight')

print("Saved plots to results/synthetic_missing_composite.pdf")
