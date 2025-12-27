# ====================================================
# ANOVA 分析 (Analysis of Variance)
# 比較不同類別間的消費金額與消費頻率差異
# ====================================================
import os
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, levene, shapiro, kruskal
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 載入共用資料模組
from data_preparation import load_and_prepare_data, get_output_folder

print("="*60)
print("📊 ANOVA 分析 (Analysis of Variance)")
print("="*60)

# 載入資料
customer_features = load_and_prepare_data()
output_folder = get_output_folder()

# ====================================================
# 1. 單因子 ANOVA：州別 vs 消費金額
# ====================================================

print("\n" + "="*60)
print("📈 1. 單因子 ANOVA：州別 vs 平均消費金額")
print("="*60)

# 取前 10 大州
top_states = customer_features['state'].value_counts().head(10).index.tolist()
state_data = customer_features[customer_features['state'].isin(top_states)].copy()

print(f"分析州別: {top_states}")
print(f"樣本數: {len(state_data):,}")

# 各組描述性統計
state_stats = state_data.groupby('state')['avg_order_value'].agg(['count', 'mean', 'std']).round(2)
state_stats.columns = ['樣本數', '平均數', '標準差']
print("\n📊 各州描述性統計:")
print(state_stats)

# Levene 檢定 (變異數同質性)
state_groups = [group['avg_order_value'].dropna() for name, group in state_data.groupby('state')]
levene_stat, levene_p = levene(*state_groups)
print(f"\n📊 Levene 檢定 (變異數同質性):")
print(f"W = {levene_stat:.4f}, p = {levene_p:.4e}")
print(f"結論: {'變異數同質' if levene_p > 0.05 else '變異數不同質 (應使用 Welch ANOVA)'}")

# 單因子 ANOVA
f_stat, p_value = f_oneway(*state_groups)
print(f"\n📊 單因子 ANOVA 結果:")
print(f"F = {f_stat:.4f}")
print(f"p-value = {p_value:.4e}")
print(f"結論: {'不同州別的消費金額有顯著差異 (p < 0.05)' if p_value < 0.05 else '不同州別的消費金額無顯著差異'}")

# 計算效果量 (Eta-squared)
ss_between = sum(len(g) * (g.mean() - state_data['avg_order_value'].mean())**2 for g in state_groups)
ss_total = sum((state_data['avg_order_value'] - state_data['avg_order_value'].mean())**2)
eta_squared = ss_between / ss_total
print(f"效果量 η² = {eta_squared:.4f}")

# Tukey HSD 事後比較
if p_value < 0.05:
    print("\n📊 Tukey HSD 事後比較:")
    tukey = pairwise_tukeyhsd(
        state_data['avg_order_value'].dropna(),
        state_data.loc[state_data['avg_order_value'].notna(), 'state']
    )
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    sig_pairs = tukey_df[tukey_df['reject'] == True]
    print(f"顯著差異的配對組數: {len(sig_pairs)}")
    if len(sig_pairs) > 0:
        print(sig_pairs.head(10))

# 視覺化
fig_state = px.box(
    state_data,
    x='state',
    y='avg_order_value',
    title=f'<b>各州平均消費金額分布</b><br><sub>ANOVA: F={f_stat:.2f}, p={p_value:.4e}, η²={eta_squared:.4f}</sub>',
    color='state'
)
fig_state.update_layout(
    xaxis_title='州別',
    yaxis_title='平均消費金額 (BRL)',
    height=500,
    template='plotly_white',
    showlegend=False
)
fig_state.update_yaxes(range=[0, 400])

output_state = os.path.join(output_folder, 'anova_state.html')
fig_state.write_html(output_state)
print(f"\n✅ 州別 ANOVA 圖已儲存: {output_state}")

# ====================================================
# 2. 單因子 ANOVA：產品類別 vs 消費金額
# ====================================================

print("\n" + "="*60)
print("📈 2. 單因子 ANOVA：產品類別 vs 平均消費金額")
print("="*60)

top_categories = customer_features['favorite_category'].value_counts().head(10).index.tolist()
cat_data = customer_features[customer_features['favorite_category'].isin(top_categories)].copy()

print(f"分析類別: {top_categories}")
print(f"樣本數: {len(cat_data):,}")

# 各組描述性統計
cat_stats = cat_data.groupby('favorite_category')['avg_order_value'].agg(['count', 'mean', 'std']).round(2)
cat_stats.columns = ['樣本數', '平均數', '標準差']
cat_stats = cat_stats.sort_values('平均數', ascending=False)
print("\n📊 各類別描述性統計:")
print(cat_stats)

# ANOVA
cat_groups = [group['avg_order_value'].dropna() for name, group in cat_data.groupby('favorite_category')]
f_stat_cat, p_value_cat = f_oneway(*cat_groups)

# 效果量
ss_between_cat = sum(len(g) * (g.mean() - cat_data['avg_order_value'].mean())**2 for g in cat_groups)
ss_total_cat = sum((cat_data['avg_order_value'] - cat_data['avg_order_value'].mean())**2)
eta_squared_cat = ss_between_cat / ss_total_cat

print(f"\n📊 單因子 ANOVA 結果:")
print(f"F = {f_stat_cat:.4f}")
print(f"p-value = {p_value_cat:.4e}")
print(f"效果量 η² = {eta_squared_cat:.4f}")
print(f"結論: {'不同產品類別的消費金額有顯著差異' if p_value_cat < 0.05 else '無顯著差異'}")

# 視覺化
cat_order = cat_stats.index.tolist()
fig_cat = px.box(
    cat_data,
    x='favorite_category',
    y='avg_order_value',
    title=f'<b>各產品類別平均消費金額分布</b><br><sub>ANOVA: F={f_stat_cat:.2f}, p={p_value_cat:.4e}, η²={eta_squared_cat:.4f}</sub>',
    category_orders={'favorite_category': cat_order},
    color='favorite_category'
)
fig_cat.update_layout(
    xaxis_title='產品類別',
    yaxis_title='平均消費金額 (BRL)',
    height=500,
    template='plotly_white',
    showlegend=False
)
fig_cat.update_xaxes(tickangle=-45)
fig_cat.update_yaxes(range=[0, 500])

output_cat = os.path.join(output_folder, 'anova_category.html')
fig_cat.write_html(output_cat)
print(f"✅ 類別 ANOVA 圖已儲存: {output_cat}")

# ====================================================
# 3. 單因子 ANOVA：付款方式 vs 消費金額
# ====================================================

print("\n" + "="*60)
print("📈 3. 單因子 ANOVA：付款方式 vs 平均消費金額")
print("="*60)

pay_stats = customer_features.groupby('preferred_payment')['avg_order_value'].agg(['count', 'mean', 'std']).round(2)
pay_stats.columns = ['樣本數', '平均數', '標準差']
print("\n📊 各付款方式描述性統計:")
print(pay_stats)

pay_groups = [group['avg_order_value'].dropna() for name, group in customer_features.groupby('preferred_payment')]
f_stat_pay, p_value_pay = f_oneway(*pay_groups)

# 效果量
ss_between_pay = sum(len(g) * (g.mean() - customer_features['avg_order_value'].mean())**2 for g in pay_groups)
ss_total_pay = sum((customer_features['avg_order_value'] - customer_features['avg_order_value'].mean())**2)
eta_squared_pay = ss_between_pay / ss_total_pay

print(f"\n📊 單因子 ANOVA 結果:")
print(f"F = {f_stat_pay:.4f}")
print(f"p-value = {p_value_pay:.4e}")
print(f"效果量 η² = {eta_squared_pay:.4f}")
print(f"結論: {'不同付款方式的消費金額有顯著差異' if p_value_pay < 0.05 else '無顯著差異'}")

# Tukey HSD
if p_value_pay < 0.05:
    print("\n📊 Tukey HSD 事後比較:")
    tukey_pay = pairwise_tukeyhsd(
        customer_features['avg_order_value'].dropna(),
        customer_features.loc[customer_features['avg_order_value'].notna(), 'preferred_payment']
    )
    print(tukey_pay)

# 視覺化
fig_pay = px.box(
    customer_features,
    x='preferred_payment',
    y='avg_order_value',
    title=f'<b>各付款方式平均消費金額分布</b><br><sub>ANOVA: F={f_stat_pay:.2f}, p={p_value_pay:.4e}, η²={eta_squared_pay:.4f}</sub>',
    color='preferred_payment'
)
fig_pay.update_layout(
    xaxis_title='付款方式',
    yaxis_title='平均消費金額 (BRL)',
    height=500,
    template='plotly_white',
    showlegend=False
)
fig_pay.update_yaxes(range=[0, 400])

output_pay = os.path.join(output_folder, 'anova_payment.html')
fig_pay.write_html(output_pay)
print(f"✅ 付款方式 ANOVA 圖已儲存: {output_pay}")

# ====================================================
# 4. 雙因子 ANOVA：回購狀態 × 付款方式
# ====================================================

print("\n" + "="*60)
print("📈 4. 雙因子 ANOVA：回購狀態 × 付款方式 對消費金額的影響")
print("="*60)

anova_data = customer_features[['avg_order_value', 'is_repeat', 'preferred_payment']].dropna().copy()
anova_data['is_repeat_label'] = anova_data['is_repeat'].map({0: '單次客', 1: '回購客'})

# 描述性統計
two_way_stats = anova_data.groupby(['is_repeat_label', 'preferred_payment'])['avg_order_value'].agg(['count', 'mean', 'std']).round(2)
print("\n📊 雙因子描述性統計:")
print(two_way_stats)

# 使用 statsmodels 進行雙因子 ANOVA
model_2way = ols('avg_order_value ~ C(is_repeat_label) * C(preferred_payment)', data=anova_data).fit()
anova_table = anova_lm(model_2way, typ=2)

print("\n📊 雙因子 ANOVA 結果 (Type II SS):")
print(anova_table)

# 主效果與交互作用解讀
print("\n🔍 結果解讀:")
for effect in ['C(is_repeat_label)', 'C(preferred_payment)', 'C(is_repeat_label):C(preferred_payment)']:
    if effect in anova_table.index:
        f_val = anova_table.loc[effect, 'F']
        p_val = anova_table.loc[effect, 'PR(>F)']
        sig = "顯著" if p_val < 0.05 else "不顯著"
        effect_name = effect.replace('C(', '').replace(')', '').replace('is_repeat_label', '回購狀態').replace('preferred_payment', '付款方式').replace(':', ' × ')
        print(f"  {effect_name}: F = {f_val:.2f}, p = {p_val:.4e} ({sig})")

# 交互作用圖
means_2way = anova_data.groupby(['is_repeat_label', 'preferred_payment'])['avg_order_value'].mean().unstack()

fig_interaction = go.Figure()
for payment in means_2way.columns:
    fig_interaction.add_trace(go.Scatter(
        x=['單次客', '回購客'],
        y=means_2way[payment].values,
        mode='lines+markers',
        name=payment,
        marker=dict(size=10)
    ))

fig_interaction.update_layout(
    title='<b>雙因子 ANOVA 交互作用圖</b><br><sub>回購狀態 × 付款方式 對消費金額的影響</sub>',
    xaxis_title='回購狀態',
    yaxis_title='平均消費金額 (BRL)',
    height=500,
    template='plotly_white',
    legend_title='付款方式'
)

output_interaction = os.path.join(output_folder, 'anova_interaction.html')
fig_interaction.write_html(output_interaction)
print(f"\n✅ 交互作用圖已儲存: {output_interaction}")

# ====================================================
# 5. Kruskal-Wallis 檢定 (非參數替代方案)
# ====================================================

print("\n" + "="*60)
print("📈 5. Kruskal-Wallis 檢定 (非參數 ANOVA)")
print("="*60)

print("\n當資料不符合常態分配或變異數同質性假設時，可使用 Kruskal-Wallis 檢定")

# 對州別進行 Kruskal-Wallis
kw_stat, kw_p = kruskal(*state_groups)
print(f"\n📊 州別 vs 消費金額 (Kruskal-Wallis):")
print(f"H = {kw_stat:.4f}")
print(f"p-value = {kw_p:.4e}")
print(f"結論: {'有顯著差異' if kw_p < 0.05 else '無顯著差異'}")

# 對類別進行 Kruskal-Wallis
kw_stat_cat, kw_p_cat = kruskal(*cat_groups)
print(f"\n📊 產品類別 vs 消費金額 (Kruskal-Wallis):")
print(f"H = {kw_stat_cat:.4f}")
print(f"p-value = {kw_p_cat:.4e}")
print(f"結論: {'有顯著差異' if kw_p_cat < 0.05 else '無顯著差異'}")

# ====================================================
# 6. ANOVA 結果彙整圖
# ====================================================

print("\n" + "="*60)
print("📊 6. 生成彙整圖表")
print("="*60)

# 彙整各 ANOVA 結果
anova_summary = pd.DataFrame({
    '分析': ['州別', '產品類別', '付款方式'],
    'F值': [f_stat, f_stat_cat, f_stat_pay],
    'p值': [p_value, p_value_cat, p_value_pay],
    '效果量 (η²)': [eta_squared, eta_squared_cat, eta_squared_pay],
    '顯著性': ['顯著' if p < 0.05 else '不顯著' for p in [p_value, p_value_cat, p_value_pay]]
})

fig_summary = make_subplots(
    rows=1, cols=2,
    subplot_titles=['F 統計量', '效果量 (η²)']
)

fig_summary.add_trace(
    go.Bar(
        x=anova_summary['分析'],
        y=anova_summary['F值'],
        marker_color=['#4ECDC4' if s == '顯著' else '#FF6B6B' for s in anova_summary['顯著性']],
        text=[f"F={f:.1f}" for f in anova_summary['F值']],
        textposition='outside',
        showlegend=False
    ),
    row=1, col=1
)

fig_summary.add_trace(
    go.Bar(
        x=anova_summary['分析'],
        y=anova_summary['效果量 (η²)'],
        marker_color=['#4ECDC4' if s == '顯著' else '#FF6B6B' for s in anova_summary['顯著性']],
        text=[f"η²={e:.3f}" for e in anova_summary['效果量 (η²)']],
        textposition='outside',
        showlegend=False
    ),
    row=1, col=2
)

fig_summary.update_layout(
    title='<b>ANOVA 分析結果彙整</b>',
    height=400,
    template='plotly_white'
)

output_summary = os.path.join(output_folder, 'anova_summary.html')
fig_summary.write_html(output_summary)
print(f"✅ ANOVA 彙整圖已儲存: {output_summary}")

# ====================================================
# 7. 分析摘要
# ====================================================

print("\n" + "="*60)
print("📋 分析摘要")
print("="*60)

summary = f"""
{'='*70}
ANOVA 分析報告
{'='*70}

📊 分析方法說明
-------------------
ANOVA (變異數分析) 用於比較三個以上組別的平均數是否有顯著差異。
- 虛無假設 (H0): 所有組別的平均數相等
- 對立假設 (H1): 至少有一組平均數與其他組不同
- 顯著水準: α = 0.05

📈 1. 單因子 ANOVA：州別 vs 平均消費金額
-------------------
- 樣本數: {len(state_data):,}
- F 統計量: {f_stat:.4f}
- p-value: {p_value:.4e}
- 效果量 η²: {eta_squared:.4f}
- 結論: {'不同州別的消費金額有顯著差異' if p_value < 0.05 else '不同州別的消費金額無顯著差異'}

📈 2. 單因子 ANOVA：產品類別 vs 平均消費金額
-------------------
- 樣本數: {len(cat_data):,}
- F 統計量: {f_stat_cat:.4f}
- p-value: {p_value_cat:.4e}
- 效果量 η²: {eta_squared_cat:.4f}
- 結論: {'不同產品類別的消費金額有顯著差異' if p_value_cat < 0.05 else '無顯著差異'}

消費金額最高的前 3 類別:
{cat_stats.head(3).to_string()}

📈 3. 單因子 ANOVA：付款方式 vs 平均消費金額
-------------------
- F 統計量: {f_stat_pay:.4f}
- p-value: {p_value_pay:.4e}
- 效果量 η²: {eta_squared_pay:.4f}
- 結論: {'不同付款方式的消費金額有顯著差異' if p_value_pay < 0.05 else '無顯著差異'}

📈 4. 雙因子 ANOVA：回購狀態 × 付款方式
-------------------
{anova_table.to_string()}

📊 效果量解釋標準 (Cohen)
-------------------
- η² < 0.01: 微小效果
- 0.01 ≤ η² < 0.06: 小效果
- 0.06 ≤ η² < 0.14: 中效果
- η² ≥ 0.14: 大效果

📁 輸出檔案
-------------------
✅ {output_state}
✅ {output_cat}
✅ {output_pay}
✅ {output_interaction}
✅ {output_summary}

{'='*70}
"""

print(summary)

# 儲存報告
report_path = os.path.join(output_folder, 'anova_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"📄 報告已儲存: {report_path}")
print("\n🎉 ANOVA 分析完成！")
