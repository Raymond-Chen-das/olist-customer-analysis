"""階段一：資料契約驗證 (Data Contract Validation).

Profiles the 9 raw Olist CSVs against a DECLARED contract — expected primary
keys, foreign-key relationships, timestamp columns — and writes the result to
docs/data-contract.md. This script only reads and reports; it does NOT
transform or load anything (that is 01_build_db.py).

Run:  ./.venv/Scripts/python.exe 00_validate_contract.py
"""
from __future__ import annotations

import sys

import pandas as pd

import config

# Windows consoles default to cp950 here and choke on emoji / Portuguese text.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NULL_FLAG_THRESHOLD = 0.20  # flag any column with >20% missing

# --- Declared contract: what each table SHOULD be --------------------------
# fk = (child_col, parent_table_stem, parent_col)
TABLES = {
    "olist_customers_dataset": dict(
        pk="customer_id", timestamps=[], fks=[],
        notes="customer_unique_id is intentionally NON-unique (one person, many orders)."),
    "olist_orders_dataset": dict(
        pk="order_id",
        timestamps=["order_purchase_timestamp", "order_approved_at",
                    "order_delivered_carrier_date", "order_delivered_customer_date",
                    "order_estimated_delivery_date"],
        fks=[("customer_id", "olist_customers_dataset", "customer_id")],
        notes=""),
    "olist_order_items_dataset": dict(
        pk=None,  # composite (order_id, order_item_id)
        timestamps=["shipping_limit_date"],
        fks=[("order_id", "olist_orders_dataset", "order_id"),
             ("product_id", "olist_products_dataset", "product_id"),
             ("seller_id", "olist_sellers_dataset", "seller_id")],
        notes="Grain = one row per item line; key is (order_id, order_item_id)."),
    "olist_order_payments_dataset": dict(
        pk=None,  # composite (order_id, payment_sequential)
        timestamps=[],
        fks=[("order_id", "olist_orders_dataset", "order_id")],
        notes="Grain = one row per payment; an order may have several payments."),
    "olist_order_reviews_dataset": dict(
        pk="review_id",
        timestamps=["review_creation_date", "review_answer_timestamp"],
        fks=[("order_id", "olist_orders_dataset", "order_id")],
        notes="review_id is known to be NOT reliably unique in the public dump."),
    "olist_products_dataset": dict(
        pk="product_id", timestamps=[],
        fks=[("product_category_name", "product_category_name_translation",
              "product_category_name")],
        notes="product_category_name has some nulls."),
    "olist_sellers_dataset": dict(
        pk="seller_id", timestamps=[], fks=[], notes=""),
    "olist_geolocation_dataset": dict(
        pk=None, timestamps=[], fks=[],
        notes="geolocation_zip_code_prefix repeats (many lat/lng per prefix)."),
    "product_category_name_translation": dict(
        pk="product_category_name", timestamps=[], fks=[],
        notes="Lookup table: PT -> EN category names."),
}


def load_all() -> dict[str, pd.DataFrame]:
    frames = {}
    for stem in TABLES:
        path = config.RAW_DIR / f"{stem}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")
        frames[stem] = pd.read_csv(path)
    return frames


def col_note(s: pd.Series) -> str:
    """One-line description of a column's value domain."""
    if pd.api.types.is_numeric_dtype(s):
        return f"range [{s.min():g}, {s.max():g}]"
    return f"{s.nunique():,} distinct"


def main() -> None:
    frames = load_all()
    md: list[str] = []
    flags: list[str] = []  # anomalies worth a human's attention

    for stem, spec in TABLES.items():
        df = frames[stem]
        md.append(f"### `{stem}` — {len(df):,} rows × {df.shape[1]} cols")
        if spec["notes"]:
            md.append(f"> {spec['notes']}")
        md.append("")
        md.append("| column | dtype | null % | value domain |")
        md.append("|---|---|---|---|")
        for c in df.columns:
            nullpct = df[c].isna().mean()
            mark = " ⚠️" if nullpct > NULL_FLAG_THRESHOLD else ""
            md.append(f"| {c} | {df[c].dtype} | {nullpct*100:.1f}%{mark} | {col_note(df[c])} |")
            if nullpct > NULL_FLAG_THRESHOLD:
                flags.append(f"[{stem}] `{c}` null {nullpct*100:.1f}% (>20%)")

        # Primary-key uniqueness
        if spec["pk"]:
            dups = int(df[spec["pk"]].duplicated().sum())
            status = "OK (unique)" if dups == 0 else f"⚠️ {dups:,} duplicate keys"
            md.append(f"\n**PK `{spec['pk']}`**: {status}")
            if dups:
                flags.append(f"[{stem}] PK `{spec['pk']}` has {dups:,} duplicates")

        # Foreign-key referential integrity
        for child_col, parent_stem, parent_col in spec["fks"]:
            child = df[child_col].dropna()
            parent_set = set(frames[parent_stem][parent_col].dropna())
            orphan = int((~child.isin(parent_set)).sum())
            opct = orphan / len(child) * 100 if len(child) else 0
            status = "OK" if orphan == 0 else f"⚠️ {orphan:,} orphans ({opct:.2f}%)"
            md.append(f"**FK `{child_col}` -> `{parent_stem}.{parent_col}`**: {status}")
            if orphan:
                flags.append(f"[{stem}] FK `{child_col}` has {orphan:,} orphans ({opct:.2f}%)")

        # Timestamp ranges
        for ts in spec["timestamps"]:
            parsed = pd.to_datetime(df[ts], errors="coerce")
            nnull = int(parsed.isna().sum())
            md.append(f"**time `{ts}`**: {parsed.min()} -> {parsed.max()}  "
                      f"(unparseable/null: {nnull:,})")
        md.append("\n---\n")

    # customer_unique_id sanity (the headline trap)
    cust = frames["olist_customers_dataset"]
    n_cid = cust["customer_id"].nunique()
    n_uid = cust["customer_unique_id"].nunique()
    md.insert(0, "## 關鍵驗證：customer_unique_id 陷阱\n"
                 f"- `customer_id` distinct: **{n_cid:,}**\n"
                 f"- `customer_unique_id` distinct: **{n_uid:,}**\n"
                 f"- 差額 {n_cid - n_uid:,} 表示同一人有多筆 customer_id → "
                 f"客戶層級分析必須用 `customer_unique_id`。\n")

    # Header + anomaly summary
    header = ["# 資料契約驗證報告 (階段一)\n",
              "由 `00_validate_contract.py` 自動產生。\n",
              f"## 異常與旗標彙總（{len(flags)} 項）\n"]
    if flags:
        header += [f"- ⚠️ {f}" for f in flags]
    else:
        header.append("- 無 >20% 缺失、無 PK 重複、無 FK 孤兒。")
    header.append("\n> 註：高缺失欄位若不參與分析則屬良性；參與前需評估。\n\n---\n")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = config.ROOT / "docs" / "data-contract.md"
    out.write_text("\n".join(header + md), encoding="utf-8")

    # Console summary
    print(f"customer_id distinct={n_cid:,}  customer_unique_id distinct={n_uid:,}")
    print(f"flags ({len(flags)}):")
    for f in flags:
        print("  [!]", f)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
