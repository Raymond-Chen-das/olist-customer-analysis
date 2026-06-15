-- 02_customer_features.sql
-- 客戶層級特徵表（grain = customer_unique_id），一律以 delivered 訂單為準。
-- 展示：多表 JOIN、window function (ROW_NUMBER)、日期運算、條件聚合。
-- 參數由 02_build_features.py 從 config.py 注入：:snapshot, :cutoff, :window
-- 單一 CREATE TABLE AS 陳述句（故可用具名參數）。

CREATE TABLE customer_features AS
WITH
-- 每張 delivered 訂單的金額 = 品項 price + 運費
order_value AS (
    SELECT oi.order_id, SUM(oi.price + oi.freight_value) AS order_revenue
    FROM order_items oi
    JOIN delivered_orders d ON d.order_id = oi.order_id
    GROUP BY oi.order_id
),
-- 每張訂單一筆評論分數（同一訂單多筆評論 -> 取平均，處理 review_id 重複）
order_review AS (
    SELECT order_id, AVG(review_score) AS review_score
    FROM order_reviews
    GROUP BY order_id
),
-- delivered 訂單明細 + 依購買時間排序（用來抓第一單、第二單）
orders_ranked AS (
    SELECT
        d.customer_unique_id,
        d.order_id,
        d.customer_state,
        d.order_purchase_timestamp,
        d.order_delivered_customer_date,
        d.order_estimated_delivery_date,
        ov.order_revenue,
        ROW_NUMBER() OVER (PARTITION BY d.customer_unique_id
                           ORDER BY d.order_purchase_timestamp, d.order_id) AS seq
    FROM delivered_orders d
    LEFT JOIN order_value ov ON ov.order_id = d.order_id
),
-- 客戶層級 R/F/M 原始值
agg AS (
    SELECT
        customer_unique_id,
        COUNT(*)                       AS frequency,
        SUM(order_revenue)             AS monetary,
        MAX(order_purchase_timestamp)  AS last_order_ts
    FROM orders_ranked
    GROUP BY customer_unique_id
),
-- 首單資訊（含首單體驗）
first_order AS (
    SELECT
        r.customer_unique_id,
        r.order_purchase_timestamp        AS first_ts,
        r.order_delivered_customer_date   AS first_delivered_ts,
        r.order_estimated_delivery_date   AS first_estimated_ts,
        r.order_revenue                   AS first_order_value,
        r.customer_state                  AS customer_state,
        rev.review_score                  AS first_review_score
    FROM orders_ranked r
    LEFT JOIN order_review rev ON rev.order_id = r.order_id
    WHERE r.seq = 1
),
-- 第二單購買時間（判斷複購窗）
second_order AS (
    SELECT customer_unique_id, order_purchase_timestamp AS second_ts
    FROM orders_ranked WHERE seq = 2
)
SELECT
    a.customer_unique_id,
    a.frequency,
    ROUND(a.monetary, 2)                                                AS monetary,
    ROUND(a.monetary / a.frequency, 2)                                  AS avg_order_value,
    CAST(julianday(:snapshot) - julianday(a.last_order_ts) AS INT)      AS recency_days,
    f.first_ts                                                          AS first_order_ts,
    a.last_order_ts,
    -- 以下為「首單當下」特徵：模塊四(複購預測)只用這些 + 客戶屬性，避免洩漏
    ROUND(f.first_order_value, 2)                                        AS first_order_value,
    f.customer_state,
    f.first_review_score,
    CAST(julianday(f.first_delivered_ts) - julianday(f.first_ts) AS INT) AS first_delivery_days,
    CASE WHEN f.first_delivered_ts IS NOT NULL
         THEN CASE WHEN julianday(f.first_delivered_ts) <= julianday(f.first_estimated_ts)
                   THEN 1 ELSE 0 END
    END                                                                 AS first_on_time,
    -- 複購標籤：僅對首單 <= cutoff 的客戶定義（保證 180 天可觀察），其餘 NULL
    CASE
        WHEN date(f.first_ts) <= :cutoff THEN
            CASE WHEN s.second_ts IS NOT NULL
                  AND (julianday(s.second_ts) - julianday(f.first_ts)) <= :window
                 THEN 1 ELSE 0 END
        ELSE NULL
    END                                                                 AS repurchase_180d
FROM agg a
JOIN first_order f  ON f.customer_unique_id = a.customer_unique_id
LEFT JOIN second_order s ON s.customer_unique_id = a.customer_unique_id;
