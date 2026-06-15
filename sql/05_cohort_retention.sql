-- 05_cohort_retention.sql — 客戶留存 cohort 分析（純 SQL）。
-- cohort = 首購月份；追蹤其後各月 offset 仍下單的客戶比例。
-- 展示：日期運算、self-join via first-order、COUNT(DISTINCT)、月份偏移。
-- 由 02_build_features.py 執行（單一 CREATE TABLE AS）。

CREATE TABLE cohort_retention AS
WITH firsts AS (
    SELECT customer_unique_id, MIN(order_purchase_timestamp) AS first_ts
    FROM delivered_orders
    GROUP BY customer_unique_id
),
orders_off AS (
    SELECT
        d.customer_unique_id,
        strftime('%Y-%m', f.first_ts) AS cohort,
        (CAST(strftime('%Y', d.order_purchase_timestamp) AS INTEGER) * 12
         + CAST(strftime('%m', d.order_purchase_timestamp) AS INTEGER))
      - (CAST(strftime('%Y', f.first_ts) AS INTEGER) * 12
         + CAST(strftime('%m', f.first_ts) AS INTEGER)) AS month_offset
    FROM delivered_orders d
    JOIN firsts f USING (customer_unique_id)
),
sizes AS (
    SELECT cohort, COUNT(DISTINCT customer_unique_id) AS cohort_size
    FROM orders_off WHERE month_offset = 0 GROUP BY cohort
)
SELECT
    o.cohort,
    o.month_offset,
    COUNT(DISTINCT o.customer_unique_id) AS active,
    s.cohort_size,
    ROUND(100.0 * COUNT(DISTINCT o.customer_unique_id) / s.cohort_size, 2) AS retention_pct
FROM orders_off o
JOIN sizes s USING (cohort)
GROUP BY o.cohort, o.month_offset, s.cohort_size;
