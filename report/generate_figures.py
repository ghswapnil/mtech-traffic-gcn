"""
Generate visualizations for the M.Tech progress report.
Creates publication-quality charts comparing model performance.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('./report/figures', exist_ok=True)

# ============================================================
# Color palette
# ============================================================
COLORS = {
    'ha': '#95a5a6',
    'arima': '#7f8c8d',
    'var': '#bdc3c7',
    'fclstm': '#3498db',
    'stgcn': '#e74c3c',
    'dcrnn': '#e67e22',
    'graphwavenet': '#9b59b6',
    'gman': '#1abc9c',
    'agcrn': '#2ecc71',
    'ttm_zero': '#f39c12',
    'ttm_gcn': '#2980b9',
    'ttm_gcn_res': '#c0392b',
}

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

# ============================================================
# Figure 1: MAE Comparison Bar Chart (All Models, 3 Horizons)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))

models = ['HA', 'ARIMA', 'FC-LSTM', 'STGCN', 'DCRNN', 'GWNet', 'GMAN', 'AGCRN',
          'TTM\n(Zero)', 'TTM+GCN\nAdapter', 'TTM+GCN\n+Res', 'Co-Fine\nTuned']

mae_15 = [4.16, 3.99, 3.44, 2.88, 2.77, 2.69, 2.80, 2.87, 3.97, 3.59, 3.58, 3.52]
mae_30 = [4.16, 5.15, 3.77, 3.47, 3.15, 3.07, 3.12, 3.23, 4.90, 4.15, 4.16, 4.07]
mae_60 = [4.16, 6.90, 4.37, 4.59, 3.60, 3.53, 3.44, 3.62, 6.20, 4.88, 4.86, 4.75]

x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, mae_15, width, label='15 min', color='#3498db', alpha=0.85)
bars2 = ax.bar(x, mae_30, width, label='30 min', color='#e67e22', alpha=0.85)
bars3 = ax.bar(x + width, mae_60, width, label='60 min', color='#e74c3c', alpha=0.85)

ax.set_ylabel('MAE (mph)')
ax.set_title('MAE Comparison Across All Models on METR-LA')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=8)
ax.legend(loc='upper left')
ax.set_ylim(0, 8)
ax.grid(axis='y', alpha=0.3)

# Highlight our models with a shaded background
ax.axvspan(7.5, 11.5, alpha=0.08, color='blue', label='_nolegend_')
ax.text(9.5, 7.5, 'Our Models', ha='center', fontsize=9, fontstyle='italic', color='#2c3e50')

plt.tight_layout()
plt.savefig('./report/figures/mae_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('./report/figures/mae_comparison.pdf', bbox_inches='tight')
print("Saved: mae_comparison.png/pdf")
plt.close()

# ============================================================
# Figure 2: RMSE Comparison Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

models_rmse = ['STGCN', 'DCRNN', 'GWNet', 'GMAN', 'AGCRN',
               'TTM\n(Zero)', 'TTM+GCN\nAdapter', 'TTM+GCN\n+Res', 'Co-Fine\nTuned']

rmse_15 = [5.74, 5.38, 5.15, 5.55, 5.54, 7.55, 6.50, 6.47, 6.57]
rmse_30 = [7.24, 6.45, 6.22, 6.49, 6.58, 9.28, 7.71, 7.63, 7.69]
rmse_60 = [9.40, 7.60, 7.37, 7.35, 7.56, 11.33, 8.98, 8.92, 8.86]

x = np.arange(len(models_rmse))

bars1 = ax.bar(x - width, rmse_15, width, label='15 min', color='#3498db', alpha=0.85)
bars2 = ax.bar(x, rmse_30, width, label='30 min', color='#e67e22', alpha=0.85)
bars3 = ax.bar(x + width, rmse_60, width, label='60 min', color='#e74c3c', alpha=0.85)

ax.set_ylabel('RMSE (mph)')
ax.set_title('RMSE Comparison on METR-LA')
ax.set_xticks(x)
ax.set_xticklabels(models_rmse, fontsize=8)
ax.legend(loc='upper left')
ax.set_ylim(0, 13)
ax.grid(axis='y', alpha=0.3)

# Highlight our models
ax.axvspan(4.5, 8.5, alpha=0.08, color='blue')
ax.text(6.5, 12, 'Our Models', ha='center', fontsize=9, fontstyle='italic', color='#2c3e50')

# Add annotation for STGCN vs Ours at 60 min
ax.annotate('Ours beats\nSTGCN', xy=(0 + width, 9.40), xytext=(1.5, 11.5),
            fontsize=8, ha='center', color='#27ae60',
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))

plt.tight_layout()
plt.savefig('./report/figures/rmse_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('./report/figures/rmse_comparison.pdf', bbox_inches='tight')
print("Saved: rmse_comparison.png/pdf")
plt.close()

# ============================================================
# Figure 3: Improvement Over Baseline (Waterfall-style)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

horizons = ['15 min', '30 min', '60 min', 'Overall']
ttm_mae = [3.97, 4.90, 6.20, 8.34]
gcn_mae = [3.59, 4.15, 4.88, 5.86]
res_mae = [3.58, 4.16, 4.86, 5.84]

x = np.arange(len(horizons))
width = 0.25

ax.bar(x - width, ttm_mae, width, label='Vanilla TTM (Zero-Shot)', color='#f39c12', alpha=0.85)
ax.bar(x, gcn_mae, width, label='TTM + GCN Adapter', color='#2980b9', alpha=0.85)
ax.bar(x + width, res_mae, width, label='TTM + GCN + Residual', color='#c0392b', alpha=0.85)

# Add percentage improvement labels
for i in range(len(horizons)):
    pct = (ttm_mae[i] - res_mae[i]) / ttm_mae[i] * 100
    ax.text(x[i] + width, res_mae[i] + 0.15, f'-{pct:.1f}%', 
            ha='center', fontsize=8, color='#c0392b', fontweight='bold')

ax.set_ylabel('MAE (mph)')
ax.set_title('Improvement Over Vanilla TTM Baseline')
ax.set_xticks(x)
ax.set_xticklabels(horizons)
ax.legend(loc='upper left')
ax.set_ylim(0, 10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('./report/figures/improvement_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('./report/figures/improvement_chart.pdf', bbox_inches='tight')
print("Saved: improvement_chart.png/pdf")
plt.close()

# ============================================================
# Figure 4: Training Loss Curves
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Without Residual
epochs_nr = list(range(1, 21))
train_loss_nr = [6.0731, 5.5348, 5.4393, 5.3745, 5.3314, 5.3079, 5.2815, 5.2679,
                 5.2521, 5.2374, 5.2301, 5.2200, 5.2140, 5.2000, 5.1940, 5.1874,
                 5.1803, 5.1739, 5.1698, 5.1605]
val_loss_nr = [5.1772, 5.0495, 4.9905, 4.9862, 4.9465, 4.9231, 4.9744, 4.9318,
               4.9139, 4.9130, 4.9179, 4.9512, 4.8934, 4.8828, 4.9027, 4.8754,
               4.8752, 4.8508, 4.9240, 4.8709]

ax1.plot(epochs_nr, train_loss_nr, 'o-', color='#3498db', markersize=4, label='Train Loss')
ax1.plot(epochs_nr, val_loss_nr, 's-', color='#e74c3c', markersize=4, label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Masked MAE Loss')
ax1.set_title('Without Residual Connection')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xlim(1, 20)

# With Residual
train_loss_r = [5.9669, 5.5913, 5.4799, 5.4235, 5.3866, 5.3545, 5.3314, 5.3114,
                5.2998, 5.2879, 5.2778, 5.2665, 5.2618, 5.2523, 5.2499, 5.2425,
                5.2317, 5.2253, 5.2211, 5.2217]
val_loss_r = [5.2564, 5.1177, 5.0460, 4.9939, 4.9661, 4.9511, 4.9707, 4.9456,
              4.9643, 4.9123, 4.9294, 4.9462, 4.9173, 4.9062, 4.9259, 4.9305,
              4.9155, 4.8697, 4.8832, 4.9176]

ax2.plot(epochs_nr, train_loss_r, 'o-', color='#3498db', markersize=4, label='Train Loss')
ax2.plot(epochs_nr, val_loss_r, 's-', color='#e74c3c', markersize=4, label='Val Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Masked MAE Loss')
ax2.set_title('With Residual Connection')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xlim(1, 20)

# Match y-axes
ymin = min(min(train_loss_nr + val_loss_nr), min(train_loss_r + val_loss_r)) - 0.2
ymax = max(max(train_loss_nr + val_loss_nr), max(train_loss_r + val_loss_r)) + 0.2
ax1.set_ylim(ymin, ymax)
ax2.set_ylim(ymin, ymax)

plt.suptitle('Training Convergence', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('./report/figures/training_curves.png', dpi=300, bbox_inches='tight')
plt.savefig('./report/figures/training_curves.pdf', bbox_inches='tight')
print("Saved: training_curves.png/pdf")
plt.close()

# ============================================================
# Figure 5: Parameter Efficiency (Bubble Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

models_p = ['STGCN', 'DCRNN', 'GWNet', 'GMAN', 'AGCRN', 'Ours\n(Adapter)', 'Ours\n(+Res)', 'Co-Fine\nTuned']
params = [1.2, 1.8, 2.5, 3.2, 0.8, 0.057, 0.057, 0.156]  # in millions
mae_60_p = [4.59, 3.60, 3.53, 3.44, 3.62, 4.88, 4.86, 4.75]
colors_p = ['#e74c3c', '#e67e22', '#9b59b6', '#1abc9c', '#2ecc71', '#2980b9', '#c0392b', '#8e44ad']

# Bubble size proportional to params
sizes = [p * 200 for p in params]

for i, model in enumerate(models_p):
    ax.scatter(params[i], mae_60_p[i], s=sizes[i], c=colors_p[i], alpha=0.7, edgecolors='black', linewidth=0.5)
    offset_y = 0.15 if i < 5 else -0.25
    ax.text(params[i], mae_60_p[i] + offset_y, model, ha='center', fontsize=8, fontweight='bold')

ax.set_xlabel('Trainable Parameters (Millions)')
ax.set_ylabel('60-min MAE (mph)')
ax.set_title('Parameter Efficiency: 60-min MAE vs. Trainable Parameters')
ax.grid(alpha=0.3)
ax.set_xlim(-0.3, 4.0)
ax.set_ylim(2.8, 5.5)

# Add annotation arrow
ax.annotate('Our models: competitive accuracy\nwith 14×-56× fewer params',
            xy=(0.057, 4.86), xytext=(1.5, 5.2),
            fontsize=9, ha='center', color='#2c3e50',
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', edgecolor='#bdc3c7'))

plt.tight_layout()
plt.savefig('./report/figures/param_efficiency.png', dpi=300, bbox_inches='tight')
plt.savefig('./report/figures/param_efficiency.pdf', bbox_inches='tight')
print("Saved: param_efficiency.png/pdf")
plt.close()

# ============================================================
# Figure 6: Horizon-wise MAE Improvement (Line Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

steps = [3, 6, 12]  # horizon steps
step_labels = ['15 min\n(step 3)', '30 min\n(step 6)', '60 min\n(step 12)']

stgcn_line = [2.88, 3.47, 4.59]
dcrnn_line = [2.77, 3.15, 3.60]
gwn_line = [2.69, 3.07, 3.53]
ttm_line = [3.97, 4.90, 6.20]
ours_line = [3.58, 4.16, 4.86]

ax.plot(steps, stgcn_line, 'D-', color='#e74c3c', markersize=7, label='STGCN', linewidth=2)
ax.plot(steps, dcrnn_line, 's-', color='#e67e22', markersize=7, label='DCRNN', linewidth=2)
ax.plot(steps, gwn_line, '^-', color='#9b59b6', markersize=7, label='Graph WaveNet', linewidth=2)
ax.plot(steps, ttm_line, 'o--', color='#f39c12', markersize=7, label='Vanilla TTM', linewidth=2, alpha=0.7)
ax.plot(steps, ours_line, 'o-', color='#2980b9', markersize=8, label='Ours (TTM+GCN+Res)', linewidth=2.5)

ax.set_xlabel('Prediction Horizon')
ax.set_ylabel('MAE (mph)')
ax.set_title('MAE Degradation Across Prediction Horizons')
ax.set_xticks(steps)
ax.set_xticklabels(step_labels)
ax.legend(loc='upper left')
ax.grid(alpha=0.3)

# Shade the region between TTM and Ours to show improvement
ax.fill_between(steps, ttm_line, ours_line, alpha=0.15, color='#2980b9', label='_nolegend_')
ax.text(9, 5.6, 'GCN Adapter\nImprovement', ha='center', fontsize=9, color='#2980b9', fontstyle='italic')

plt.tight_layout()
plt.savefig('./report/figures/horizon_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('./report/figures/horizon_comparison.pdf', bbox_inches='tight')
print("Saved: horizon_comparison.png/pdf")
plt.close()

print("\n✅ All 6 figures generated in ./report/figures/")
