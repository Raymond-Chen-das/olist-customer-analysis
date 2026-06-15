-- 01_schema.sql — 索引與 delivered 視圖（由 01_build_db.py 載入資料後執行）
-- 決策(2026-06-15)：僅 order_status='delivered' 視為有效訂單。

-- 連接鍵索引（讓後續多表 JOIN 走得動）
CREATE INDEX IF NOT EXISTS ix_customers_cid ON customers(customer_id);
CREATE INDEX IF NOT EXISTS ix_customers_uid ON customers(customer_unique_id);
CREATE INDEX IF NOT EXISTS ix_orders_oid    ON orders(order_id);
CREATE INDEX IF NOT EXISTS ix_orders_cid    ON orders(customer_id);
CREATE INDEX IF NOT EXISTS ix_orders_status ON orders(order_status);
CREATE INDEX IF NOT EXISTS ix_items_oid     ON order_items(order_id);
CREATE INDEX IF NOT EXISTS ix_items_pid     ON order_items(product_id);
CREATE INDEX IF NOT EXISTS ix_items_sid     ON order_items(seller_id);
CREATE INDEX IF NOT EXISTS ix_payments_oid  ON order_payments(order_id);
CREATE INDEX IF NOT EXISTS ix_reviews_oid   ON order_reviews(order_id);
CREATE INDEX IF NOT EXISTS ix_products_pid  ON products(product_id);

-- 有效訂單骨幹：delivered 訂單 + 掛上 customer_unique_id（客戶層級分析的起點）
DROP VIEW IF EXISTS delivered_orders;
CREATE VIEW delivered_orders AS
SELECT
    o.order_id,
    c.customer_unique_id,
    o.customer_id,
    c.customer_state,
    c.customer_city,
    c.customer_zip_code_prefix,
    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_status = 'delivered';
