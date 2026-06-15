"""階段一→二 橋接：把 9 張原始 CSV 載入 SQLite，建立索引與 delivered 視圖。

只負責「忠實載入 + 基礎結構」，不做分析性轉換（RFM / cohort 等屬模塊一的 SQL，
於後續腳本以 sql/ 內的腳本完成）。

Run:  ./.venv/Scripts/python.exe 01_build_db.py
"""
from __future__ import annotations

import sqlite3
import sys

import pandas as pd

import config

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 原始檔名 -> 乾淨資料表名
TABLE_MAP = {
    "olist_customers_dataset": "customers",
    "olist_orders_dataset": "orders",
    "olist_order_items_dataset": "order_items",
    "olist_order_payments_dataset": "order_payments",
    "olist_order_reviews_dataset": "order_reviews",
    "olist_products_dataset": "products",
    "olist_sellers_dataset": "sellers",
    "olist_geolocation_dataset": "geolocation",
    "product_category_name_translation": "category_translation",
}


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()  # rebuild from scratch for reproducibility

    con = sqlite3.connect(config.DB_PATH)
    for stem, table in TABLE_MAP.items():
        df = pd.read_csv(config.RAW_DIR / f"{stem}.csv")
        df.to_sql(table, con, if_exists="replace", index=False)
        print(f"loaded {table:20s} {len(df):>9,} rows")

    # 索引 + delivered 視圖（SQL 與程式碼分離）
    schema_sql = (config.SQL_DIR / "01_schema.sql").read_text(encoding="utf-8")
    con.executescript(schema_sql)

    # 驗證：delivered 視圖筆數應為 96,478
    cur = con.cursor()
    n_deliv = cur.execute("SELECT COUNT(*) FROM delivered_orders").fetchone()[0]
    n_uid = cur.execute(
        "SELECT COUNT(DISTINCT customer_unique_id) FROM delivered_orders").fetchone()[0]
    print(f"\ndelivered_orders rows:                {n_deliv:,}")
    print(f"distinct customer_unique_id (delivered): {n_uid:,}")

    con.commit()
    con.close()
    print(f"\nwrote {config.DB_PATH}")


if __name__ == "__main__":
    main()
