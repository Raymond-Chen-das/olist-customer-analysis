# ====================================================
# 邏輯斯回歸分析 (Logistic Regression)
# 預測顧客是否會回購
# ====================================================
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 載入共用資料模組
from data_preparation import load_and_prepare_data, get_output_folder

print("="*60)
print("📊 邏輯斯回歸分析 (Logistic Regression)")
print("="*60)

# 載入資料
customer_features = load_and_prepare_data()
output_folder = get_output_folder()

# ====================================================
# 1. 資料準備
# ====================================================

print("\n" + "="*60)
print("📈 1. 資料準備")
print("="*60)

# 選擇預測變項
X_cols = ['avg_delay_days', 'avg_freight_ratio', 'avg_review_score',
          'avg_payment_installments', 'avg_order_value']

X = customer_features[X_cols].copy()
y = customer_features['is_repeat'].copy()

# 移除遺漏值
valid_idx = ~(X.isna().any(axis=1) | y.isna())
X = X[valid_idx]
y = y[valid_idx]

print(f"有效樣本數: {len(y):,}")
print(f"依變項: 是否回購 (is_repeat)")
print(f"  - 單次客 (0): {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
print(f"  - 回購客 (1): {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
print(f"自變項: {X_cols}")

# ====================================================
# 2. 使用 Statsmodels 建立邏輯斯回歸
# ====================================================

print("\n" + "="*60)
print("📈 2. 邏輯斯回歸模型 (Statsmodels)")
print("="*60)

# 加入常數項
X_const = sm.add_constant(X)

# 建立模型
model_logit = sm.Logit(y, X_const).fit()

print("\n" + "="*50)
print("邏輯斯回歸結果摘要")
print("="*50)
print(model_logit.summary())

# ====================================================
# 3. 回歸係數與勝算比 (Odds Ratio)
# ====================================================

print("\n" + "="*60)
print("📊 3. 回歸係數與勝算比解讀")
print("="*60)

# 計算勝算比與信賴區間
odds_ratios = np.exp(model_logit.params)
conf_int = model_logit.conf_int()
or_ci = np.exp(conf_int)

logit_results = pd.DataFrame({
    '變項': ['截距 (Intercept)'] + X_cols,
    '係數 (B)': model_logit.params.values,
    '標準誤 (SE)': model_logit.bse.values,
    'Wald χ²': model_logit.tvalues.values ** 2,
    'p 值': model_logit.pvalues.values,
    '勝算比 (OR)': odds_ratios.values,
    'OR 95% CI 下界': or_ci[0].values,
    'OR 95% CI 上界': or_ci[1].values
})

logit_results['顯著性'] = logit_results['p 值'].apply(
    lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
)

print("\n📊 邏輯斯回歸係數與勝算比:")
print(logit_results.to_string(index=False))

print("\n🔍 顯著預測變項解讀:")
sig_vars = logit_results[(logit_results['顯著性'] != '') & (logit_results['變項'] != '截距 (Intercept)')]
for _, row in sig_vars.iterrows():
    or_val = row['勝算比 (OR)']
    if or_val > 1:
        change = (or_val - 1) * 100
        direction = "增加"
    else:
        change = (1 - or_val) * 100
        direction = "減少"
    print(f"  - {row['變項']}: OR = {or_val:.3f}")
    print(f"    每增加 1 單位，回購的勝算{direction} {change:.1f}%")

# ====================================================
# 4. 模型配適度檢驗
# ====================================================

print("\n" + "="*60)
print("📈 4. 模型配適度檢驗")
print("="*60)

# Pseudo R-squared
print(f"McFadden's Pseudo R²: {model_logit.prsquared:.4f}")

# Log-Likelihood
print(f"Log-Likelihood: {model_logit.llf:.2f}")
print(f"Log-Likelihood (Null): {model_logit.llnull:.2f}")

# Likelihood Ratio Test
from scipy import stats
lr_stat = -2 * (model_logit.llnull - model_logit.llf)
lr_pvalue = 1 - stats.chi2.cdf(lr_stat, len(X_cols))
print(f"Likelihood Ratio χ²: {lr_stat:.2f}, p = {lr_pvalue:.4e}")

# AIC & BIC
print(f"AIC: {model_logit.aic:.2f}")
print(f"BIC: {model_logit.bic:.2f}")

# ====================================================
# 5. 使用 sklearn 進行預測評估
# ====================================================

print("\n" + "="*60)
print("📈 5. 模型預測評估 (sklearn)")
print("="*60)

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 建立模型 (使用 class_weight='balanced' 處理不平衡)
lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

# 預測
y_pred = lr_model.predict(X_test)
y_pred_proba = lr_model.predict_proba(X_test)[:, 1]

# 分類報告
print("\n📊 分類報告:")
print(classification_report(y_test, y_pred, target_names=['單次客', '回購客']))

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
print("\n📊 混淆矩陣:")
print(cm)

# 交叉驗證
cv_scores = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"\n📊 5-Fold 交叉驗證 ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")

# ====================================================
# 6. ROC 曲線與 AUC
# ====================================================

print("\n" + "="*60)
print("📊 6. ROC 曲線分析")
print("="*60)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"AUC (Area Under Curve): {roc_auc:.4f}")

# 找最佳閾值 (Youden's J)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
print(f"最佳閾值 (Youden's J): {best_threshold:.4f}")
print(f"  - 敏感度 (Sensitivity): {tpr[best_idx]:.4f}")
print(f"  - 特異度 (Specificity): {1-fpr[best_idx]:.4f}")

# ====================================================
# 7. 視覺化
# ====================================================

print("\n" + "="*60)
print("📊 7. 生成視覺化圖表")
print("="*60)

# 7.1 勝算比森林圖
or_plot = logit_results[logit_results['變項'] != '截距 (Intercept)'].copy()
or_plot['變項中文'] = ['延遲天數', '運費比例', '評論分數', '分期期數', '消費金額']

fig_or = go.Figure()

# 添加信賴區間線
for i, row in or_plot.iterrows():
    fig_or.add_trace(go.Scatter(
        x=[row['OR 95% CI 下界'], row['OR 95% CI 上界']],
        y=[row['變項中文'], row['變項中文']],
        mode='lines',
        line=dict(color='#4ECDC4', width=3),
        showlegend=False
    ))

# 添加勝算比點
fig_or.add_trace(go.Scatter(
    x=or_plot['勝算比 (OR)'],
    y=or_plot['變項中文'],
    mode='markers+text',
    marker=dict(size=12, color='#4ECDC4'),
    text=[f"{or_:.3f}{sig}" for or_, sig in zip(or_plot['勝算比 (OR)'], or_plot['顯著性'])],
    textposition='top center',
    showlegend=False
))

# 參考線 (OR = 1)
fig_or.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="OR=1")

fig_or.update_layout(
    title='<b>邏輯斯回歸勝算比 (Odds Ratio) 森林圖</b><br><sub>OR > 1 表示增加回購機率，OR < 1 表示降低回購機率</sub>',
    xaxis_title='勝算比 (95% CI)',
    yaxis_title='',
    height=450,
    template='plotly_white',
    xaxis=dict(type='log')  # 對數刻度
)

output_or = os.path.join(output_folder, 'logistic_odds_ratio.html')
fig_or.write_html(output_or)
print(f"✅ 勝算比圖已儲存: {output_or}")

# 7.2 ROC 曲線
fig_roc = go.Figure()

fig_roc.add_trace(go.Scatter(
    x=fpr, y=tpr,
    mode='lines',
    name=f'ROC (AUC = {roc_auc:.3f})',
    line=dict(color='#4ECDC4', width=2)
))

fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='隨機猜測',
    line=dict(color='gray', dash='dash')
))

# 標記最佳閾值點
fig_roc.add_trace(go.Scatter(
    x=[fpr[best_idx]],
    y=[tpr[best_idx]],
    mode='markers',
    name=f'最佳閾值 ({best_threshold:.3f})',
    marker=dict(size=12, color='red', symbol='star')
))

fig_roc.update_layout(
    title='<b>ROC 曲線 (Receiver Operating Characteristic)</b>',
    xaxis_title='1 - 特異度 (False Positive Rate)',
    yaxis_title='敏感度 (True Positive Rate)',
    height=500,
    template='plotly_white',
    legend=dict(x=0.6, y=0.2)
)

output_roc = os.path.join(output_folder, 'logistic_roc_curve.html')
fig_roc.write_html(output_roc)
print(f"✅ ROC 曲線已儲存: {output_roc}")

# 7.3 混淆矩陣熱力圖
fig_cm = go.Figure(data=go.Heatmap(
    z=cm,
    x=['預測: 單次客', '預測: 回購客'],
    y=['實際: 單次客', '實際: 回購客'],
    colorscale='Blues',
    text=cm,
    texttemplate='%{text}',
    textfont={"size": 20}
))

fig_cm.update_layout(
    title='<b>混淆矩陣</b>',
    xaxis_title='預測值',
    yaxis_title='實際值',
    height=400,
    template='plotly_white'
)

output_cm = os.path.join(output_folder, 'logistic_confusion_matrix.html')
fig_cm.write_html(output_cm)
print(f"✅ 混淆矩陣已儲存: {output_cm}")

# 7.4 Precision-Recall 曲線
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

fig_pr = go.Figure()

fig_pr.add_trace(go.Scatter(
    x=recall, y=precision,
    mode='lines',
    name=f'PR Curve (AUC = {pr_auc:.3f})',
    line=dict(color='#FF6B6B', width=2)
))

fig_pr.update_layout(
    title='<b>Precision-Recall 曲線</b>',
    xaxis_title='Recall (敏感度)',
    yaxis_title='Precision (精確度)',
    height=500,
    template='plotly_white'
)

output_pr = os.path.join(output_folder, 'logistic_pr_curve.html')
fig_pr.write_html(output_pr)
print(f"✅ PR 曲線已儲存: {output_pr}")

# ====================================================
# 8. 卡方檢定 (類別變項)
# ====================================================

print("\n" + "="*60)
print("📊 8. 卡方獨立性檢定")
print("="*60)

# 回購狀態 vs 付款方式
print("\n📈 8.1 回購狀態 × 付款方式:")
contingency_pay = pd.crosstab(customer_features['is_repeat'], customer_features['preferred_payment'])
chi2_pay, p_pay, dof_pay, expected_pay = chi2_contingency(contingency_pay)

print(f"卡方值 χ² = {chi2_pay:.4f}")
print(f"自由度 df = {dof_pay}")
print(f"p-value = {p_pay:.4e}")
print(f"結論: {'回購狀態與付款方式有顯著關聯' if p_pay < 0.05 else '無顯著關聯'}")

# Cramér's V (效果量)
n = contingency_pay.sum().sum()
min_dim = min(contingency_pay.shape) - 1
cramers_v = np.sqrt(chi2_pay / (n * min_dim))
print(f"Cramér's V = {cramers_v:.4f}")

print("\n交叉表:")
print(contingency_pay)

# ====================================================
# 9. 與決策樹的比較
# ====================================================

print("\n" + "="*60)
print("📊 9. 邏輯斯回歸 vs 決策樹比較")
print("="*60)

print("""
📋 方法比較:

| 特性           | 邏輯斯回歸                | 決策樹                    |
|----------------|---------------------------|---------------------------|
| 模型類型       | 參數模型                  | 非參數模型                |
| 可解釋性       | 係數與勝算比              | 規則與分支                |
| 處理非線性     | 需要手動添加交互項        | 自動捕捉                  |
| 過擬合風險     | 較低                      | 較高（需剪枝）            |
| 特徵重要性     | 透過係數/OR               | 透過 Gini/資訊增益        |
| 假設           | 線性關係                  | 無假設                    |
| 適用情境       | 需要解釋因果關係          | 需要規則導向決策          |
""")

# ====================================================
# 10. 分析摘要
# ====================================================

print("\n" + "="*60)
print("📋 分析摘要")
print("="*60)

summary = f"""
{'='*70}
邏輯斯回歸分析報告
{'='*70}

📊 研究問題
-------------------
探討哪些因素會影響顧客是否回購？

📈 模型設定
-------------------
依變項 (Y): 是否回購 (0=單次客, 1=回購客)
自變項 (X):
  - 平均延遲天數 (avg_delay_days)
  - 平均運費比例 (avg_freight_ratio)
  - 平均評論分數 (avg_review_score)
  - 平均分期期數 (avg_payment_installments)
  - 平均消費金額 (avg_order_value)

樣本數: {len(y):,}
  - 單次客: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)
  - 回購客: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)

📊 模型配適度
-------------------
McFadden's Pseudo R²: {model_logit.prsquared:.4f}
AIC: {model_logit.aic:.2f}
BIC: {model_logit.bic:.2f}

📈 回歸係數與勝算比
-------------------
{logit_results.to_string(index=False)}

📊 預測評估結果
-------------------
ROC-AUC: {roc_auc:.4f}
5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})
最佳閾值: {best_threshold:.4f}

混淆矩陣:
{cm}

📊 卡方檢定結果
-------------------
回購狀態 × 付款方式:
χ² = {chi2_pay:.2f}, df = {dof_pay}, p = {p_pay:.4e}
Cramér's V = {cramers_v:.4f}
結論: {'顯著關聯' if p_pay < 0.05 else '無顯著關聯'}

🔍 主要發現
-------------------
"""

# 加入顯著變項的解讀
for _, row in sig_vars.iterrows():
    or_val = row['勝算比 (OR)']
    if or_val > 1:
        summary += f"- {row['變項']}: OR={or_val:.3f}，每增加1單位，回購勝算增加 {(or_val-1)*100:.1f}%\n"
    else:
        summary += f"- {row['變項']}: OR={or_val:.3f}，每增加1單位，回購勝算減少 {(1-or_val)*100:.1f}%\n"

summary += f"""
📁 輸出檔案
-------------------
✅ {output_or}
✅ {output_roc}
✅ {output_cm}
✅ {output_pr}

{'='*70}
"""

print(summary)

# 儲存報告
report_path = os.path.join(output_folder, 'logistic_regression_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"📄 報告已儲存: {report_path}")
print("\n🎉 邏輯斯回歸分析完成！")
