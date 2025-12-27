# ====================================================
# Olist 電商回購分析完整版
# ====================================================
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Plotly 視覺化
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 機器學習
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 繁體中文字型
matplotlib.rcParams['axes.unicode_minus'] = False

# ====================================================
# Part 1: 資料載入與特徵工程
# ====================================================

# 修正版路徑設定
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

if os.path.basename(script_dir) == 'raw':
    raw_folder = script_dir
else:
    raw_folder = os.path.join(script_dir, 'raw')

# 建立輸出資料夾
output_folder = os.path.join(script_dir, 'output')
os.makedirs(output_folder, exist_ok=True)

print(f"📁 當前執行路徑: {script_dir}")
print(f"📁 資料夾路徑: {raw_folder}")
print(f"📁 圖片輸出路徑: {output_folder}")

# 載入資料集
orders = pd.read_csv(os.path.join(raw_folder, 'olist_orders_dataset.csv'))
order_items = pd.read_csv(os.path.join(raw_folder, 'olist_order_items_dataset.csv'))
customers = pd.read_csv(os.path.join(raw_folder, 'olist_customers_dataset.csv'))
reviews = pd.read_csv(os.path.join(raw_folder, 'olist_order_reviews_dataset.csv'))
payments = pd.read_csv(os.path.join(raw_folder, 'olist_order_payments_dataset.csv'))
products = pd.read_csv(os.path.join(raw_folder, 'olist_products_dataset.csv'))
product_translation = pd.read_csv(os.path.join(raw_folder, 'product_category_name_translation.csv'))

print(f"✅ 資料載入完成")
print(f"訂單數: {len(orders)}, 顧客數: {len(customers)}, 評論數: {len(reviews)}")

# 資料合併與預處理
df = orders.merge(customers, on='customer_id', how='left')

order_summary = order_items.groupby('order_id').agg({
    'price': 'sum',
    'freight_value': 'sum'
}).reset_index()
order_summary.columns = ['order_id', 'total_price', 'total_freight']
df = df.merge(order_summary, on='order_id', how='left')

review_summary = reviews.groupby('order_id').agg({
    'review_score': 'mean'
}).reset_index()
df = df.merge(review_summary, on='order_id', how='left')

payment_summary = payments.groupby('order_id').agg({
    'payment_installments': 'mean',
    'payment_value': 'sum'
}).reset_index()
df = df.merge(payment_summary, on='order_id', how='left')

order_products = order_items.merge(products, on='product_id', how='left')
order_products = order_products.merge(product_translation, on='product_category_name', how='left')

top_category = order_products.groupby('order_id')['product_category_name_english'].agg(
    lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
).reset_index()
top_category.columns = ['order_id', 'top_category']
df = df.merge(top_category, on='order_id', how='left')

# 特徵工程
date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 
             'order_estimated_delivery_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df['delay_days'] = (df['order_delivered_customer_date'] - 
                     df['order_estimated_delivery_date']).dt.days
df['freight_ratio'] = df['total_freight'] / (df['total_price'] + df['total_freight'] + 1e-6)

df_completed = df[df['order_status'] == 'delivered'].copy()
print(f"✅ 已完成訂單數: {len(df_completed)}")

# 顧客層級聚合
customer_features = df_completed.groupby('customer_unique_id').agg({
    'order_id': 'count',
    'delay_days': 'mean',
    'freight_ratio': 'mean',
    'review_score': 'mean',
    'payment_installments': 'mean',
    'total_price': 'mean',
    'top_category': lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
}).reset_index()

customer_features.columns = ['customer_unique_id', 'frequency', 'avg_delay_days', 
                             'avg_freight_ratio', 'avg_review_score', 
                             'avg_payment_installments', 'avg_order_value', 'favorite_category']

customer_features['is_repeat'] = (customer_features['frequency'] > 1).astype(int)
customer_features['avg_delay_days'].fillna(0, inplace=True)
customer_features['avg_review_score'].fillna(customer_features['avg_review_score'].median(), inplace=True)

print(f"\n📊 顧客層級特徵表")
print(customer_features.head())
print(f"\n回購客比例: {customer_features['is_repeat'].mean():.2%}")
print(f"單次客數量: {(customer_features['is_repeat']==0).sum()}")
print(f"回購客數量: {(customer_features['is_repeat']==1).sum()}")

# ====================================================
# Part 2: 視覺化分析（Plotly）
# ====================================================

print("\n" + "="*60)
print("📊 開始生成視覺化圖表...")
print("="*60)

# 圖表 1: 回購客比例分布
fig1 = go.Figure()

repeat_counts = customer_features['is_repeat'].value_counts().sort_index()
labels = ['單次購買客', '回購客']
colors = ['#FF6B6B', '#4ECDC4']

fig1.add_trace(go.Bar(
    x=labels,
    y=repeat_counts.values,
    text=[f'{v:,}<br>({v/len(customer_features)*100:.1f}%)' for v in repeat_counts.values],
    textposition='outside',
    marker_color=colors,
    hovertemplate='<b>%{x}</b><br>數量: %{y:,}<extra></extra>'
))

fig1.update_layout(
    title='<b>顧客回購分布：97% 僅購買一次</b>',
    xaxis_title='顧客類型',
    yaxis_title='顧客數量',
    font=dict(size=14),
    height=500,
    template='plotly_white',
    showlegend=False
)

output_path1 = os.path.join(output_folder, '01_回購比例分布.html')
fig1.write_html(output_path1)
print(f"✅ 圖表 1 已儲存: {output_path1}")

# 圖表 2: 回購客 vs 單次客的關鍵指標對比
comparison_metrics = customer_features.groupby('is_repeat').agg({
    'avg_delay_days': 'mean',
    'avg_freight_ratio': 'mean',
    'avg_review_score': 'mean',
    'avg_payment_installments': 'mean',
    'avg_order_value': 'mean'
}).T

comparison_metrics.columns = ['單次客', '回購客']
comparison_metrics['差異(%)'] = ((comparison_metrics['回購客'] - comparison_metrics['單次客']) / 
                                 comparison_metrics['單次客'] * 100)

fig2 = make_subplots(
    rows=2, cols=3,
    subplot_titles=('平均延遲天數', '平均運費比例', '平均評論分數', 
                    '平均分期期數', '平均訂單金額', '指標差異比較'),
    specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
)

metrics = ['avg_delay_days', 'avg_freight_ratio', 'avg_review_score', 
           'avg_payment_installments', 'avg_order_value']
positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]

for metric, pos in zip(metrics, positions):
    values = comparison_metrics.loc[metric, ['單次客', '回購客']]
    fig2.add_trace(
        go.Bar(x=['單次客', '回購客'], y=values, 
               marker_color=['#FF6B6B', '#4ECDC4'],
               showlegend=False,
               text=[f'{v:.2f}' for v in values],
               textposition='outside'),
        row=pos[0], col=pos[1]
    )

# 差異比較圖
fig2.add_trace(
    go.Bar(x=comparison_metrics.index, 
           y=comparison_metrics['差異(%)'],
           marker_color=['#FF6B6B' if x < 0 else '#4ECDC4' for x in comparison_metrics['差異(%)']],
           showlegend=False,
           text=[f'{v:+.1f}%' for v in comparison_metrics['差異(%)']],
           textposition='outside'),
    row=2, col=3
)

fig2.update_layout(height=700, title_text="<b>回購客 vs 單次客：關鍵指標對比</b>", 
                   template='plotly_white')
fig2.update_xaxes(tickangle=-45, row=2, col=3)

output_path2 = os.path.join(output_folder, '02_關鍵指標對比.html')
fig2.write_html(output_path2)
print(f"✅ 圖表 2 已儲存: {output_path2}")

# ====================================================
# Part 3: 決策樹建模
# ====================================================

print("\n" + "="*60)
print("🤖 開始建立決策樹模型...")
print("="*60)

# 準備建模資料
feature_cols = ['avg_delay_days', 'avg_freight_ratio', 'avg_review_score', 
                'avg_payment_installments', 'avg_order_value']

X = customer_features[feature_cols].copy()
y = customer_features['is_repeat'].copy()

# 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 建立決策樹（處理不平衡資料）
dt_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=100,
    min_samples_leaf=50,
    class_weight='balanced',  # 處理不平衡
    random_state=42
)

dt_model.fit(X_train, y_train)

# 預測與評估
y_pred = dt_model.predict(X_test)

print("\n📊 模型評估結果:")
print(classification_report(y_test, y_pred, target_names=['單次客', '回購客']))

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
print(f"\n混淆矩陣:\n{cm}")

# 圖表 3: 特徵重要性
feature_importance = pd.DataFrame({
    '特徵': ['平均延遲天數', '平均運費比例', '平均評論分數', '平均分期期數', '平均訂單金額'],
    '重要性': dt_model.feature_importances_
}).sort_values('重要性', ascending=True)

fig3 = go.Figure(go.Bar(
    x=feature_importance['重要性'],
    y=feature_importance['特徵'],
    orientation='h',
    marker_color='#4ECDC4',
    text=[f'{v:.3f}' for v in feature_importance['重要性']],
    textposition='outside'
))

fig3.update_layout(
    title='<b>影響回購的關鍵因素排名</b>',
    xaxis_title='特徵重要性',
    yaxis_title='',
    height=400,
    template='plotly_white'
)

output_path3 = os.path.join(output_folder, '03_特徵重要性.html')
fig3.write_html(output_path3)
print(f"✅ 圖表 3 已儲存: {output_path3}")

# 圖表 4: 決策樹視覺化（使用 matplotlib）
plt.figure(figsize=(20, 10))
plot_tree(dt_model, 
          feature_names=['平均延遲天數', '平均運費比例', '平均評論分數', '平均分期期數', '平均訂單金額'],
          class_names=['單次客', '回購客'],
          filled=True,
          rounded=True,
          fontsize=10)

plt.title('決策樹結構：回購預測模型', fontsize=16, pad=20)
output_path4 = os.path.join(output_folder, '04_決策樹結構.png')
plt.savefig(output_path4, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 圖表 4 已儲存: {output_path4}")

# ====================================================
# 匯出分析結果摘要
# ====================================================

summary_report = f"""
{'='*60}
Olist 電商回購分析報告摘要
{'='*60}

📊 資料概況
-------------------
- 總顧客數: {len(customer_features):,}
- 單次購買客: {(customer_features['is_repeat']==0).sum():,} ({(customer_features['is_repeat']==0).sum()/len(customer_features)*100:.1f}%)
- 回購客: {(customer_features['is_repeat']==1).sum():,} ({customer_features['is_repeat'].mean()*100:.1f}%)

🔍 關鍵發現
-------------------
1. 平均評論分數差異: {comparison_metrics.loc['avg_review_score', '差異(%)']:.1f}%
   → 回購客平均評分: {comparison_metrics.loc['avg_review_score', '回購客']:.2f}
   → 單次客平均評分: {comparison_metrics.loc['avg_review_score', '單次客']:.2f}

2. 物流延遲差異: {comparison_metrics.loc['avg_delay_days', '差異(%)']:.1f}%
   → 回購客平均延遲: {comparison_metrics.loc['avg_delay_days', '回購客']:.2f} 天
   → 單次客平均延遲: {comparison_metrics.loc['avg_delay_days', '單次客']:.2f} 天

3. 運費比例差異: {comparison_metrics.loc['avg_freight_ratio', '差異(%)']:.1f}%

🎯 決策樹模型結果
-------------------
最重要的 3 個特徵:
{feature_importance.tail(3).to_string(index=False)}

📁 輸出檔案
-------------------
✅ {output_path1}
✅ {output_path2}
✅ {output_path3}
✅ {output_path4}

{'='*60}
"""

print(summary_report)

# 儲存摘要報告
report_path = os.path.join(output_folder, '分析報告摘要.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(f"\n📄 完整報告已儲存: {report_path}")
print(f"\n🎉 所有分析完成！請查看 {output_folder} 資料夾")