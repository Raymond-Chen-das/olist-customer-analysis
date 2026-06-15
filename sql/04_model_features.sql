-- 04_model_features.sql — 模塊四（複購挖掘）建模特徵表。
-- 只含「首單當下 + 客戶屬性」特徵 + 標籤，避免洩漏；僅建模集（label 非 NULL）。
-- B 階段以 SQL 擴充首單特徵：品項數、運費比、購買月份。
-- 由 05_propensity.py 執行（單一 CREATE TABLE AS）。

CREATE TABLE model_features AS
WITH ranked AS (
    SELECT d.customer_unique_id, d.order_id, d.order_purchase_timestamp,
           ROW_NUMBER() OVER (PARTITION BY d.customer_unique_id
                              ORDER BY d.order_purchase_timestamp, d.order_id) AS seq
    FROM delivered_orders d
),
first_o AS (
    SELECT customer_unique_id, order_id, order_purchase_timestamp
    FROM ranked WHERE seq = 1
),
-- 首單主品類（首件）
first_cat AS (
    SELECT f.customer_unique_id,
           COALESCE(t.product_category_name_english, p.product_category_name, 'unknown')
               AS first_category
    FROM first_o f
    JOIN order_items oi ON oi.order_id = f.order_id AND oi.order_item_id = 1
    LEFT JOIN products p ON p.product_id = oi.product_id
    LEFT JOIN category_translation t ON t.product_category_name = p.product_category_name
),
-- 首單品項數與金額拆解（運費 vs 商品）
first_agg AS (
    SELECT f.customer_unique_id,
           COUNT(*)               AS first_item_count,
           SUM(oi.freight_value)  AS first_freight,
           SUM(oi.price)          AS first_price
    FROM first_o f
    JOIN order_items oi ON oi.order_id = f.order_id
    GROUP BY f.customer_unique_id
)
SELECT
    cf.customer_unique_id,
    cf.repurchase_180d,
    cf.first_order_value,
    cf.first_review_score,
    cf.first_delivery_days,
    cf.first_on_time,
    cf.customer_state,
    fc.first_category,
    fa.first_item_count,
    ROUND(fa.first_freight / NULLIF(fa.first_price + fa.first_freight, 0), 4)
        AS first_freight_ratio,
    CAST(strftime('%m', cf.first_order_ts) AS INTEGER) AS first_order_month
FROM customer_features cf
JOIN first_cat fc ON fc.customer_unique_id = cf.customer_unique_id
JOIN first_agg fa ON fa.customer_unique_id = cf.customer_unique_id
WHERE cf.repurchase_180d IS NOT NULL;
