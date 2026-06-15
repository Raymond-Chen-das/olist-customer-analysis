# 資料契約驗證報告 (階段一)

由 `00_validate_contract.py` 自動產生。

## 異常與旗標彙總（4 項）

- ⚠️ [olist_order_reviews_dataset] `review_comment_title` null 88.3% (>20%)
- ⚠️ [olist_order_reviews_dataset] `review_comment_message` null 58.2% (>20%)
- ⚠️ [olist_order_reviews_dataset] PK `review_id` has 827 duplicates
- ⚠️ [olist_products_dataset] FK `product_category_name` has 13 orphans (0.04%)

> 註：高缺失欄位若不參與分析則屬良性；參與前需評估。

---

## 關鍵驗證：customer_unique_id 陷阱
- `customer_id` distinct: **99,441**
- `customer_unique_id` distinct: **96,096**
- 差額 3,345 表示同一人有多筆 customer_id → 客戶層級分析必須用 `customer_unique_id`。

### `olist_customers_dataset` — 99,441 rows × 5 cols
> customer_unique_id is intentionally NON-unique (one person, many orders).

| column | dtype | null % | value domain |
|---|---|---|---|
| customer_id | str | 0.0% | 99,441 distinct |
| customer_unique_id | str | 0.0% | 96,096 distinct |
| customer_zip_code_prefix | int64 | 0.0% | range [1003, 99990] |
| customer_city | str | 0.0% | 4,119 distinct |
| customer_state | str | 0.0% | 27 distinct |

**PK `customer_id`**: OK (unique)

---

### `olist_orders_dataset` — 99,441 rows × 8 cols

| column | dtype | null % | value domain |
|---|---|---|---|
| order_id | str | 0.0% | 99,441 distinct |
| customer_id | str | 0.0% | 99,441 distinct |
| order_status | str | 0.0% | 8 distinct |
| order_purchase_timestamp | str | 0.0% | 98,875 distinct |
| order_approved_at | str | 0.2% | 90,733 distinct |
| order_delivered_carrier_date | str | 1.8% | 81,018 distinct |
| order_delivered_customer_date | str | 3.0% | 95,664 distinct |
| order_estimated_delivery_date | str | 0.0% | 459 distinct |

**PK `order_id`**: OK (unique)
**FK `customer_id` -> `olist_customers_dataset.customer_id`**: OK
**time `order_purchase_timestamp`**: 2016-09-04 21:15:19 -> 2018-10-17 17:30:18  (unparseable/null: 0)
**time `order_approved_at`**: 2016-09-15 12:16:38 -> 2018-09-03 17:40:06  (unparseable/null: 160)
**time `order_delivered_carrier_date`**: 2016-10-08 10:34:01 -> 2018-09-11 19:48:28  (unparseable/null: 1,783)
**time `order_delivered_customer_date`**: 2016-10-11 13:46:32 -> 2018-10-17 13:22:46  (unparseable/null: 2,965)
**time `order_estimated_delivery_date`**: 2016-09-30 00:00:00 -> 2018-11-12 00:00:00  (unparseable/null: 0)

---

### `olist_order_items_dataset` — 112,650 rows × 7 cols
> Grain = one row per item line; key is (order_id, order_item_id).

| column | dtype | null % | value domain |
|---|---|---|---|
| order_id | str | 0.0% | 98,666 distinct |
| order_item_id | int64 | 0.0% | range [1, 21] |
| product_id | str | 0.0% | 32,951 distinct |
| seller_id | str | 0.0% | 3,095 distinct |
| shipping_limit_date | str | 0.0% | 93,318 distinct |
| price | float64 | 0.0% | range [0.85, 6735] |
| freight_value | float64 | 0.0% | range [0, 409.68] |
**FK `order_id` -> `olist_orders_dataset.order_id`**: OK
**FK `product_id` -> `olist_products_dataset.product_id`**: OK
**FK `seller_id` -> `olist_sellers_dataset.seller_id`**: OK
**time `shipping_limit_date`**: 2016-09-19 00:15:34 -> 2020-04-09 22:35:08  (unparseable/null: 0)

---

### `olist_order_payments_dataset` — 103,886 rows × 5 cols
> Grain = one row per payment; an order may have several payments.

| column | dtype | null % | value domain |
|---|---|---|---|
| order_id | str | 0.0% | 99,440 distinct |
| payment_sequential | int64 | 0.0% | range [1, 29] |
| payment_type | str | 0.0% | 5 distinct |
| payment_installments | int64 | 0.0% | range [0, 24] |
| payment_value | float64 | 0.0% | range [0, 13664.1] |
**FK `order_id` -> `olist_orders_dataset.order_id`**: OK

---

### `olist_order_reviews_dataset` — 100,000 rows × 7 cols
> review_id is known to be NOT reliably unique in the public dump.

| column | dtype | null % | value domain |
|---|---|---|---|
| review_id | str | 0.0% | 99,173 distinct |
| order_id | str | 0.0% | 99,441 distinct |
| review_score | int64 | 0.0% | range [1, 5] |
| review_comment_title | str | 88.3% ⚠️ | 4,600 distinct |
| review_comment_message | str | 58.2% ⚠️ | 36,921 distinct |
| review_creation_date | str | 0.0% | 637 distinct |
| review_answer_timestamp | str | 0.0% | 99,010 distinct |

**PK `review_id`**: ⚠️ 827 duplicate keys
**FK `order_id` -> `olist_orders_dataset.order_id`**: OK
**time `review_creation_date`**: 2016-10-02 00:00:00 -> 2018-08-31 00:00:00  (unparseable/null: 0)
**time `review_answer_timestamp`**: 2016-10-07 18:32:28 -> 2018-10-29 12:27:35  (unparseable/null: 0)

---

### `olist_products_dataset` — 32,951 rows × 9 cols
> product_category_name has some nulls.

| column | dtype | null % | value domain |
|---|---|---|---|
| product_id | str | 0.0% | 32,951 distinct |
| product_category_name | str | 1.9% | 73 distinct |
| product_name_lenght | float64 | 1.9% | range [5, 76] |
| product_description_lenght | float64 | 1.9% | range [4, 3992] |
| product_photos_qty | float64 | 1.9% | range [1, 20] |
| product_weight_g | float64 | 0.0% | range [0, 40425] |
| product_length_cm | float64 | 0.0% | range [7, 105] |
| product_height_cm | float64 | 0.0% | range [2, 105] |
| product_width_cm | float64 | 0.0% | range [6, 118] |

**PK `product_id`**: OK (unique)
**FK `product_category_name` -> `product_category_name_translation.product_category_name`**: ⚠️ 13 orphans (0.04%)

---

### `olist_sellers_dataset` — 3,095 rows × 4 cols

| column | dtype | null % | value domain |
|---|---|---|---|
| seller_id | str | 0.0% | 3,095 distinct |
| seller_zip_code_prefix | int64 | 0.0% | range [1001, 99730] |
| seller_city | str | 0.0% | 611 distinct |
| seller_state | str | 0.0% | 23 distinct |

**PK `seller_id`**: OK (unique)

---

### `olist_geolocation_dataset` — 1,000,163 rows × 5 cols
> geolocation_zip_code_prefix repeats (many lat/lng per prefix).

| column | dtype | null % | value domain |
|---|---|---|---|
| geolocation_zip_code_prefix | int64 | 0.0% | range [1001, 99990] |
| geolocation_lat | float64 | 0.0% | range [-36.6054, 45.0659] |
| geolocation_lng | float64 | 0.0% | range [-101.467, 121.105] |
| geolocation_city | str | 0.0% | 8,011 distinct |
| geolocation_state | str | 0.0% | 27 distinct |

---

### `product_category_name_translation` — 71 rows × 2 cols
> Lookup table: PT -> EN category names.

| column | dtype | null % | value domain |
|---|---|---|---|
| product_category_name | str | 0.0% | 71 distinct |
| product_category_name_english | str | 0.0% | 71 distinct |

**PK `product_category_name`**: OK (unique)

---
