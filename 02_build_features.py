"""模塊一核心：用純 SQL 建立 customer_features 表，並回報健全性檢查。

執行 sql/02_customer_features.sql（日期/窗長參數由 config.py 注入），
然後印出 frequency 分布、複購率、以及 180 天標籤的正樣本數
（直接回答「180 天窗正樣本是否足夠」這個待確認假設）。

Run:  ./.venv/Scripts/python.exe 02_build_features.py
"""
from __future__ import annotations

import sqlite3
import sys

import config

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> None:
    con = sqlite3.connect(config.DB_PATH)
    cur = con.cursor()

    sql = (config.SQL_DIR / "02_customer_features.sql").read_text(encoding="utf-8")
    cur.execute("DROP TABLE IF EXISTS customer_features")
    cur.execute(sql, {
        "snapshot": config.SNAPSHOT_DATE.strftime("%Y-%m-%d"),
        "cutoff": config.MODEL_FIRST_ORDER_CUTOFF.strftime("%Y-%m-%d"),
        "window": config.REPURCHASE_WINDOW_DAYS,
    })
    # cohort 留存表（純 SQL）
    cohort_sql = (config.SQL_DIR / "05_cohort_retention.sql").read_text(encoding="utf-8")
    cur.execute("DROP TABLE IF EXISTS cohort_retention")
    cur.executescript(cohort_sql)
    con.commit()

    n = cur.execute("SELECT COUNT(*) FROM customer_features").fetchone()[0]
    print(f"customer_features rows: {n:,}")

    # frequency 分布
    print("\nfrequency 分布 (delivered 訂單數):")
    for freq, cnt in cur.execute(
        "SELECT CASE WHEN frequency>=4 THEN '4+' ELSE CAST(frequency AS TEXT) END AS f,"
        " COUNT(*) FROM customer_features GROUP BY f ORDER BY f").fetchall():
        print(f"  {freq:>3} 單: {cnt:>7,} ({cnt/n*100:5.2f}%)")
    repeat = cur.execute(
        "SELECT COUNT(*) FROM customer_features WHERE frequency>=2").fetchone()[0]
    print(f"  -> 複購客戶 (>=2 單): {repeat:,} ({repeat/n*100:.2f}%)")

    # 180 天複購標籤（建模集：label 非 NULL）
    model_n, pos = cur.execute(
        "SELECT COUNT(*), SUM(repurchase_180d) FROM customer_features"
        " WHERE repurchase_180d IS NOT NULL").fetchone()
    print(f"\n180天複購標籤建模集: {model_n:,} 位（首單<=cutoff）")
    print(f"  正樣本 (180天內複購): {pos:,} ({pos/model_n*100:.2f}%)")
    print(f"  負樣本: {model_n-pos:,}")

    # 健全性：null 檢查
    print("\nnull 檢查:")
    for col in ["monetary", "first_order_value", "first_review_score",
                "first_delivery_days", "first_on_time"]:
        nnull = cur.execute(
            f"SELECT COUNT(*) FROM customer_features WHERE {col} IS NULL").fetchone()[0]
        print(f"  {col:20s} null: {nnull:,} ({nnull/n*100:.2f}%)")

    con.close()


if __name__ == "__main__":
    main()
