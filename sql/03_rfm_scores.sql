-- 03_rfm_scores.sql — 傳統 RFM 五分箱評分。
-- R/M 用 NTILE(5)；F 因 97% 客戶=1（退化），改用實際次數規則評分，
-- 避免 NTILE 把大量 frequency=1 的 ties 任意切到不同箱而失真。
-- 由 04_rfm.py 執行（單一 CREATE TABLE AS）。

CREATE TABLE rfm_scores AS
WITH scored AS (
    SELECT
        customer_unique_id,
        recency_days,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,  -- 越近期 → 分越高
        NTILE(5) OVER (ORDER BY monetary ASC)      AS m_score,  -- 金額越高 → 分越高
        CASE frequency WHEN 1 THEN 1 WHEN 2 THEN 3 WHEN 3 THEN 4 ELSE 5 END AS f_score
    FROM customer_features
)
SELECT
    s.*,
    CASE
        WHEN r_score >= 4 AND m_score >= 4 THEN '高價值 (Champions)'
        WHEN r_score <= 2 AND m_score >= 4 THEN '高額流失風險 (At Risk)'
        WHEN r_score >= 3 AND m_score >= 3 THEN '忠誠潛力 (Loyal)'
        WHEN r_score >= 4 AND m_score <= 2 THEN '新客待培養 (New)'
        WHEN r_score <= 2 AND m_score <= 2 THEN '沉睡 (Hibernating)'
        ELSE '一般 (Regular)'
    END AS rfm_segment
FROM scored s;
