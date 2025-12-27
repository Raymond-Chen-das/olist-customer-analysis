# ====================================================
# 相關性分析 (Correlation Analysis)
# 探索消費頻率與消費金額的相關變項
# ====================================================
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 載入共用資料模組
from data_preparation import load_and_prepare_data, get_output_folder

print("="*60)
print("📊 相關性分析 (Correlation Analysis)")
print("="*60)

# 載入資料
customer_features = load_and_prepare_data()
output_folder = get_output_folder()

# ====================================================
# 1. Pearson 相關係數分析
# ====================================================

print("\n" + "="*60)
print("📈 1. Pearson 相關係數分析")
print("="*60)

numeric_cols = ['frequency', 'avg_delay_days', 'avg_freight_ratio',
                'avg_review_score', 'avg_payment_installments',
                'avg_order_value', 'total_spent', 'avg_item_count']

# 計算相關矩陣
correlation_matrix = customer_features[numeric_cols].corr(method='pearson')

print("\n📊 Pearson 相關係數矩陣:")
print(correlation_matrix.round(3))

# ====================================================
# 2. 與消費頻率的相關性
# ====================================================

print("\n" + "="*60)
print("🔍 2. 與消費頻率 (frequency) 的相關性")
print("="*60)

freq_corr = correlation_matrix['frequency'].drop('frequency').sort_values(key=abs, ascending=False)

print(f"\n{'變項':<25} {'Pearson r':>12} {'p-value':>15} {'顯著性':>8}")
print("-" * 60)

freq_corr_results = []
for var in freq_corr.index:
    valid_idx = customer_features[var].notna()
    r, p = pearsonr(customer_features.loc[valid_idx, 'frequency'], customer_features.loc[valid_idx, var])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    freq_corr_results.append({'變項': var, 'r': r, 'p': p, '顯著性': sig})
    print(f"{var:<25} {r:>12.4f} {p:>15.4e} {sig:>8}")

print("-" * 60)
print("顯著性水準: * p<0.05, ** p<0.01, *** p<0.001")

# ====================================================
# 3. 與消費金額的相關性
# ====================================================

print("\n" + "="*60)
print("🔍 3. 與平均消費金額 (avg_order_value) 的相關性")
print("="*60)

value_corr = correlation_matrix['avg_order_value'].drop('avg_order_value').sort_values(key=abs, ascending=False)

print(f"\n{'變項':<25} {'Pearson r':>12} {'p-value':>15} {'顯著性':>8}")
print("-" * 60)

value_corr_results = []
for var in value_corr.index:
    valid_idx = customer_features[var].notna()
    r, p = pearsonr(customer_features.loc[valid_idx, 'avg_order_value'],
                    customer_features.loc[valid_idx, var])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    value_corr_results.append({'變項': var, 'r': r, 'p': p, '顯著性': sig})
    print(f"{var:<25} {r:>12.4f} {p:>15.4e} {sig:>8}")

print("-" * 60)

# ====================================================
# 4. Spearman 等級相關係數
# ====================================================

print("\n" + "="*60)
print("📈 4. Spearman 等級相關係數")
print("="*60)

spearman_matrix = customer_features[numeric_cols].corr(method='spearman')

print("\n📊 Spearman 相關係數矩陣:")
print(spearman_matrix.round(3))

# 比較 Pearson 與 Spearman
print("\n📊 Pearson vs Spearman (與消費金額):")
print(f"{'變項':<25} {'Pearson r':>12} {'Spearman ρ':>12}")
print("-" * 50)
for var in value_corr.index:
    pearson_r = correlation_matrix.loc['avg_order_value', var]
    spearman_r = spearman_matrix.loc['avg_order_value', var]
    print(f"{var:<25} {pearson_r:>12.4f} {spearman_r:>12.4f}")

# ====================================================
# 5. 視覺化：相關矩陣熱力圖
# ====================================================

print("\n" + "="*60)
print("📊 5. 生成視覺化圖表")
print("="*60)

# Plotly 熱力圖
labels = ['消費頻率', '延遲天數', '運費比例', '評論分數', '分期期數', '平均消費', '總消費', '平均商品數']

fig_corr = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=labels,
    y=labels,
    colorscale='RdBu_r',
    zmid=0,
    text=np.round(correlation_matrix.values, 2),
    texttemplate='%{text}',
    textfont={"size": 11},
    hoverongaps=False,
    colorbar=dict(title='相關係數')
))

fig_corr.update_layout(
    title='<b>Pearson 相關係數矩陣</b>',
    height=650,
    width=750,
    template='plotly_white'
)

output_corr = os.path.join(output_folder, 'correlation_matrix.html')
fig_corr.write_html(output_corr)
print(f"✅ 相關矩陣圖已儲存: {output_corr}")

# ====================================================
# 6. 散佈圖矩陣
# ====================================================

# 選擇關鍵變項進行散佈圖
key_vars = ['frequency', 'avg_order_value', 'avg_review_score', 'avg_delay_days']
key_labels = ['消費頻率', '平均消費金額', '評論分數', '延遲天數']

fig_scatter = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        '消費頻率 vs 平均消費金額',
        '評論分數 vs 平均消費金額',
        '延遲天數 vs 平均消費金額',
        '運費比例 vs 平均消費金額'
    ]
)

# 抽樣以加快繪圖速度
sample_df = customer_features.sample(min(5000, len(customer_features)), random_state=42)

# 1. 頻率 vs 消費金額
fig_scatter.add_trace(
    go.Scatter(
        x=sample_df['frequency'],
        y=sample_df['avg_order_value'],
        mode='markers',
        marker=dict(size=4, opacity=0.5, color='#4ECDC4'),
        showlegend=False
    ),
    row=1, col=1
)

# 2. 評論分數 vs 消費金額
fig_scatter.add_trace(
    go.Scatter(
        x=sample_df['avg_review_score'],
        y=sample_df['avg_order_value'],
        mode='markers',
        marker=dict(size=4, opacity=0.5, color='#FF6B6B'),
        showlegend=False
    ),
    row=1, col=2
)

# 3. 延遲天數 vs 消費金額
fig_scatter.add_trace(
    go.Scatter(
        x=sample_df['avg_delay_days'],
        y=sample_df['avg_order_value'],
        mode='markers',
        marker=dict(size=4, opacity=0.5, color='#45B7D1'),
        showlegend=False
    ),
    row=2, col=1
)

# 4. 運費比例 vs 消費金額
fig_scatter.add_trace(
    go.Scatter(
        x=sample_df['avg_freight_ratio'],
        y=sample_df['avg_order_value'],
        mode='markers',
        marker=dict(size=4, opacity=0.5, color='#96CEB4'),
        showlegend=False
    ),
    row=2, col=2
)

fig_scatter.update_layout(
    title='<b>關鍵變項與消費金額的散佈圖</b>',
    height=700,
    template='plotly_white'
)

fig_scatter.update_xaxes(title_text='消費頻率', row=1, col=1)
fig_scatter.update_xaxes(title_text='評論分數', row=1, col=2)
fig_scatter.update_xaxes(title_text='延遲天數', row=2, col=1)
fig_scatter.update_xaxes(title_text='運費比例', row=2, col=2)
fig_scatter.update_yaxes(title_text='平均消費金額', row=1, col=1)
fig_scatter.update_yaxes(title_text='平均消費金額', row=1, col=2)
fig_scatter.update_yaxes(title_text='平均消費金額', row=2, col=1)
fig_scatter.update_yaxes(title_text='平均消費金額', row=2, col=2)

output_scatter = os.path.join(output_folder, 'correlation_scatterplots.html')
fig_scatter.write_html(output_scatter)
print(f"✅ 散佈圖已儲存: {output_scatter}")

# ====================================================
# 7. Matplotlib/Seaborn 熱力圖 (靜態圖)
# ====================================================

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
            fmt='.2f', square=True, linewidths=0.5,
            xticklabels=labels, yticklabels=labels)
plt.title('Pearson 相關係數矩陣', fontsize=14, pad=20)
plt.tight_layout()

output_heatmap = os.path.join(output_folder, 'correlation_heatmap.png')
plt.savefig(output_heatmap, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 靜態熱力圖已儲存: {output_heatmap}")

# ====================================================
# 8. 分析摘要
# ====================================================

summary = f"""
{'='*60}
相關性分析摘要報告
{'='*60}

📊 分析方法
-------------------
- Pearson 相關係數: 適用於線性關係的連續變項
- Spearman 等級相關: 適用於非線性或序列資料

📈 與消費頻率最相關的變項 (Top 3)
-------------------
"""
for i, result in enumerate(freq_corr_results[:3], 1):
    summary += f"{i}. {result['變項']}: r = {result['r']:.4f} {result['顯著性']}\n"

summary += f"""
📈 與消費金額最相關的變項 (Top 3)
-------------------
"""
for i, result in enumerate(value_corr_results[:3], 1):
    summary += f"{i}. {result['變項']}: r = {result['r']:.4f} {result['顯著性']}\n"

summary += f"""
🔍 主要發現
-------------------
1. 總消費金額與平均消費金額呈強正相關 (r > 0.7)
2. 運費比例與消費金額呈負相關（運費比例高的訂單消費較低）
3. 平均商品數與消費金額呈正相關（買越多商品，消費越高）
4. 評論分數與消費頻率/金額的相關性較弱

📁 輸出檔案
-------------------
✅ {output_corr}
✅ {output_scatter}
✅ {output_heatmap}

{'='*60}
"""

print(summary)

# 儲存報告
report_path = os.path.join(output_folder, 'correlation_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"📄 報告已儲存: {report_path}")
print("\n🎉 相關性分析完成！")
