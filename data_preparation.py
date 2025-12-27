# ====================================================
# 資料準備共用模組
# 供各統計分析檔案載入使用
# ====================================================
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    載入並準備 Olist 電商資料
    返回顧客層級的特徵資料框
    """
    # 路徑設定
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

    if os.path.basename(script_dir) == 'raw':
        raw_folder = script_dir
    else:
        raw_folder = os.path.join(script_dir, 'raw')

    print(f"📁 資料夾路徑: {raw_folder}")

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
        'freight_value': 'sum',
        'order_item_id': 'count'
    }).reset_index()
    order_summary.columns = ['order_id', 'total_price', 'total_freight', 'item_count']
    df = df.merge(order_summary, on='order_id', how='left')

    review_summary = reviews.groupby('order_id').agg({
        'review_score': 'mean'
    }).reset_index()
    df = df.merge(review_summary, on='order_id', how='left')

    payment_summary = payments.groupby('order_id').agg({
        'payment_installments': 'mean',
        'payment_value': 'sum',
        'payment_type': lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
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
        'total_price': ['mean', 'sum'],
        'item_count': 'mean',
        'top_category': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
        'customer_state': 'first',
        'payment_type': lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
    }).reset_index()

    customer_features.columns = ['customer_unique_id', 'frequency', 'avg_delay_days',
                                 'avg_freight_ratio', 'avg_review_score',
                                 'avg_payment_installments', 'avg_order_value', 'total_spent',
                                 'avg_item_count', 'favorite_category', 'state', 'preferred_payment']

    customer_features['is_repeat'] = (customer_features['frequency'] > 1).astype(int)
    customer_features['avg_delay_days'].fillna(0, inplace=True)
    customer_features['avg_review_score'].fillna(customer_features['avg_review_score'].median(), inplace=True)

    print(f"\n📊 顧客層級特徵表已建立")
    print(f"總顧客數: {len(customer_features):,}")
    print(f"回購客比例: {customer_features['is_repeat'].mean():.2%}")

    return customer_features

def get_output_folder():
    """取得輸出資料夾路徑"""
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    output_folder = os.path.join(script_dir, 'output_stats')
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

if __name__ == "__main__":
    # 測試資料載入
    df = load_and_prepare_data()
    print("\n資料欄位:")
    print(df.columns.tolist())
    print("\n資料預覽:")
    print(df.head())
