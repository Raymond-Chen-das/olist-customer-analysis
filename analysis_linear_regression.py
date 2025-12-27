# ====================================================
# 多元線性回歸分析 (Multiple Linear Regression)
# 探討影響消費金額的多變量因素
# ====================================================
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 載入共用資料模組
from data_preparation import load_and_prepare_data, get_output_folder

print("="*60)
print("📊 多元線性回歸分析 (Multiple Linear Regression)")
print("="*60)

# 載入資料
customer_features = load_and_prepare_data()
output_folder = get_output_folder()

# ====================================================
# 1. 準備資料
# ====================================================

print("\n" + "="*60)
print("📈 1. 資料準備")
print("="*60)

# 選擇預測變項
X_cols = ['avg_delay_days', 'avg_freight_ratio', 'avg_review_score',
          'avg_payment_installments', 'avg_item_count']

X = customer_features[X_cols].copy()
y = customer_features['avg_order_value'].copy()

# 移除遺漏值
valid_idx = ~(X.isna().any(axis=1) | y.isna())
X = X[valid_idx]
y = y[valid_idx]

print(f"有效樣本數: {len(y):,}")
print(f"依變項: 平均消費金額 (avg_order_value)")
print(f"自變項: {X_cols}")

# ====================================================
# 2. 檢查多元共線性 (VIF)
# ====================================================

print("\n" + "="*60)
print("📈 2. 多元共線性檢測 (Variance Inflation Factor)")
print("="*60)

X_with_const = sm.add_constant(X)

vif_data = pd.DataFrame()
vif_data['變項'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X_with_const.values, i+1) for i in range(len(X.columns))]

print("\n📊 VIF 值 (VIF > 10 表示嚴重共線性):")
print(vif_data.to_string(index=False))

if (vif_data['VIF'] > 10).any():
    print("\n⚠️ 警告: 存在嚴重多元共線性問題！")
else:
    print("\n✅ 無嚴重多元共線性問題")

# ====================================================
# 3. 建立 OLS 回歸模型
# ====================================================

print("\n" + "="*60)
print("📈 3. 多元線性回歸模型 (OLS)")
print("="*60)

# 加入常數項
X_const = sm.add_constant(X)

# 建立模型
model_ols = sm.OLS(y, X_const).fit()

print("\n" + "="*50)
print("OLS 回歸結果摘要")
print("="*50)
print(model_ols.summary())

# ====================================================
# 4. 回歸係數解讀
# ====================================================

print("\n" + "="*60)
print("📊 4. 回歸係數解讀")
print("="*60)

coef_df = pd.DataFrame({
    '變項': ['截距 (Intercept)'] + X_cols,
    '係數 (B)': model_ols.params.values,
    '標準誤 (SE)': model_ols.bse.values,
    't 值': model_ols.tvalues.values,
    'p 值': model_ols.pvalues.values,
    '95% CI 下界': model_ols.conf_int()[0].values,
    '95% CI 上界': model_ols.conf_int()[1].values
})

coef_df['顯著性'] = coef_df['p 值'].apply(
    lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
)

# 計算標準化係數 (Beta)
X_std = (X - X.mean()) / X.std()
X_std_const = sm.add_constant(X_std)
model_std = sm.OLS((y - y.mean()) / y.std(), X_std_const).fit()
coef_df['標準化係數 (β)'] = [np.nan] + list(model_std.params.values[1:])

print("\n📊 回歸係數表:")
print(coef_df.to_string(index=False))

print(f"\n📈 模型配適度:")
print(f"R² = {model_ols.rsquared:.4f}")
print(f"Adjusted R² = {model_ols.rsquared_adj:.4f}")
print(f"F-statistic = {model_ols.fvalue:.2f}")
print(f"F-test p-value = {model_ols.f_pvalue:.4e}")

# ====================================================
# 5. 殘差診斷
# ====================================================

print("\n" + "="*60)
print("📈 5. 殘差診斷")
print("="*60)

# 預測值與殘差
y_pred = model_ols.predict(X_const)
residuals = model_ols.resid

print(f"殘差平均值: {residuals.mean():.6f} (應接近 0)")
print(f"殘差標準差: {residuals.std():.4f}")

# Durbin-Watson 檢定 (自相關)
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuals)
print(f"Durbin-Watson 統計量: {dw:.4f} (接近 2 表示無自相關)")

# ====================================================
# 6. 模型預測能力評估
# ====================================================

print("\n" + "="*60)
print("📈 6. 模型預測能力評估 (交叉驗證)")
print("="*60)

# 分割訓練/測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# sklearn 線性回歸
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_test = lr_model.predict(X_test)

# 評估指標
r2_test = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)

print(f"訓練集 R²: {lr_model.score(X_train, y_train):.4f}")
print(f"測試集 R²: {r2_test:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# ====================================================
# 7. 視覺化
# ====================================================

print("\n" + "="*60)
print("📊 7. 生成視覺化圖表")
print("="*60)

# 7.1 回歸係數圖
coef_plot = coef_df[coef_df['變項'] != '截距 (Intercept)'].copy()
coef_plot['變項中文'] = ['延遲天數', '運費比例', '評論分數', '分期期數', '平均商品數']

fig_coef = go.Figure()

# 添加係數與信賴區間
fig_coef.add_trace(go.Bar(
    x=coef_plot['變項中文'],
    y=coef_plot['係數 (B)'],
    marker_color=['#FF6B6B' if c < 0 else '#4ECDC4' for c in coef_plot['係數 (B)']],
    text=[f"{c:.2f}{s}" for c, s in zip(coef_plot['係數 (B)'], coef_plot['顯著性'])],
    textposition='outside',
    error_y=dict(
        type='data',
        symmetric=False,
        array=coef_plot['95% CI 上界'] - coef_plot['係數 (B)'],
        arrayminus=coef_plot['係數 (B)'] - coef_plot['95% CI 下界']
    )
))

fig_coef.update_layout(
    title=f'<b>線性回歸係數 (依變項：平均消費金額)</b><br><sub>R² = {model_ols.rsquared:.4f}, Adj. R² = {model_ols.rsquared_adj:.4f}</sub>',
    xaxis_title='預測變項',
    yaxis_title='回歸係數 (B)',
    height=500,
    template='plotly_white'
)

output_coef = os.path.join(output_folder, 'regression_coefficients.html')
fig_coef.write_html(output_coef)
print(f"✅ 回歸係數圖已儲存: {output_coef}")

# 7.2 標準化係數圖
fig_beta = go.Figure()

beta_values = coef_plot['標準化係數 (β)'].values
fig_beta.add_trace(go.Bar(
    x=coef_plot['變項中文'],
    y=beta_values,
    marker_color=['#FF6B6B' if c < 0 else '#4ECDC4' for c in beta_values],
    text=[f"β={b:.3f}{s}" for b, s in zip(beta_values, coef_plot['顯著性'])],
    textposition='outside'
))

fig_beta.update_layout(
    title='<b>標準化回歸係數 (Beta)</b><br><sub>可比較各變項的相對重要性</sub>',
    xaxis_title='預測變項',
    yaxis_title='標準化係數 (β)',
    height=500,
    template='plotly_white'
)

output_beta = os.path.join(output_folder, 'regression_beta.html')
fig_beta.write_html(output_beta)
print(f"✅ 標準化係數圖已儲存: {output_beta}")

# 7.3 殘差診斷圖
fig_resid = make_subplots(
    rows=2, cols=2,
    subplot_titles=['殘差 vs 預測值', '殘差直方圖', 'Q-Q 圖', '殘差分布']
)

# 殘差 vs 預測值
sample_idx = np.random.choice(len(y_pred), min(5000, len(y_pred)), replace=False)
fig_resid.add_trace(
    go.Scatter(
        x=y_pred.iloc[sample_idx] if hasattr(y_pred, 'iloc') else y_pred[sample_idx],
        y=residuals.iloc[sample_idx] if hasattr(residuals, 'iloc') else residuals[sample_idx],
        mode='markers',
        marker=dict(size=4, opacity=0.5, color='#4ECDC4'),
        showlegend=False
    ),
    row=1, col=1
)
fig_resid.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

# 殘差直方圖
fig_resid.add_trace(
    go.Histogram(x=residuals, nbinsx=50, marker_color='#4ECDC4', showlegend=False),
    row=1, col=2
)

# Q-Q 圖
from scipy import stats
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
fig_resid.add_trace(
    go.Scatter(x=osm, y=osr, mode='markers', marker=dict(size=4, color='#4ECDC4'), showlegend=False),
    row=2, col=1
)
fig_resid.add_trace(
    go.Scatter(x=osm, y=slope*np.array(osm)+intercept, mode='lines',
               line=dict(color='red', dash='dash'), showlegend=False),
    row=2, col=1
)

# 殘差分布 (Box plot)
fig_resid.add_trace(
    go.Box(y=residuals, marker_color='#4ECDC4', showlegend=False),
    row=2, col=2
)

fig_resid.update_layout(
    title='<b>殘差診斷圖</b>',
    height=700,
    template='plotly_white'
)

fig_resid.update_xaxes(title_text='預測值', row=1, col=1)
fig_resid.update_yaxes(title_text='殘差', row=1, col=1)
fig_resid.update_xaxes(title_text='殘差', row=1, col=2)
fig_resid.update_yaxes(title_text='頻率', row=1, col=2)
fig_resid.update_xaxes(title_text='理論分位數', row=2, col=1)
fig_resid.update_yaxes(title_text='樣本分位數', row=2, col=1)

output_resid = os.path.join(output_folder, 'regression_diagnostics.html')
fig_resid.write_html(output_resid)
print(f"✅ 殘差診斷圖已儲存: {output_resid}")

# ====================================================
# 8. 分析摘要
# ====================================================

print("\n" + "="*60)
print("📋 分析摘要")
print("="*60)

summary = f"""
{'='*70}
多元線性回歸分析報告
{'='*70}

📊 研究問題
-------------------
探討哪些因素會影響顧客的平均消費金額？

📈 模型設定
-------------------
依變項 (Y): 平均消費金額 (avg_order_value)
自變項 (X):
  - 平均延遲天數 (avg_delay_days)
  - 平均運費比例 (avg_freight_ratio)
  - 平均評論分數 (avg_review_score)
  - 平均分期期數 (avg_payment_installments)
  - 平均商品數量 (avg_item_count)

樣本數: {len(y):,}

📊 模型配適度
-------------------
R² = {model_ols.rsquared:.4f} (解釋 {model_ols.rsquared*100:.1f}% 的變異)
Adjusted R² = {model_ols.rsquared_adj:.4f}
F-statistic = {model_ols.fvalue:.2f}, p-value = {model_ols.f_pvalue:.4e}

📈 回歸係數結果
-------------------
{coef_df.to_string(index=False)}

🔍 顯著預測變項解讀
-------------------
"""

# 找出顯著的預測變項
sig_vars = coef_df[(coef_df['顯著性'] != '') & (coef_df['變項'] != '截距 (Intercept)')]
for _, row in sig_vars.iterrows():
    direction = "正向" if row['係數 (B)'] > 0 else "負向"
    summary += f"- {row['變項']}: {direction}影響 (B = {row['係數 (B)']:.3f}, p < 0.05)\n"

summary += f"""
📊 交叉驗證結果
-------------------
訓練集 R²: {lr_model.score(X_train, y_train):.4f}
測試集 R²: {r2_test:.4f}
RMSE: {rmse:.2f}
MAE: {mae:.2f}

📊 多元共線性檢測 (VIF)
-------------------
{vif_data.to_string(index=False)}

📁 輸出檔案
-------------------
✅ {output_coef}
✅ {output_beta}
✅ {output_resid}

{'='*70}
"""

print(summary)

# 儲存報告
report_path = os.path.join(output_folder, 'regression_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"📄 報告已儲存: {report_path}")
print("\n🎉 多元線性回歸分析完成！")
