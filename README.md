# Olist 電商顧客回購行為分析

本專案針對巴西最大電商平台 Olist 的公開數據進行深度分析，探索影響顧客消費金額與回購行為的關鍵因素，並提供可行的商業策略建議。

## 專案背景

Olist 是巴西的電商整合平台，連接小型企業與主要電商市場。本專案使用 2016-2018 年的真實交易數據，分析以下核心問題：

- **為何只有 3% 的顧客會回購？**
- 哪些因素顯著影響顧客的消費金額？
- 如何識別高價值顧客並進行精準行銷？

## 數據概覽

| 指標 | 數值 |
|------|------|
| 總顧客數 | 93,358 人 |
| 單次購買客 | 90,557 人 (97.0%) |
| 回購客 | 2,801 人 (3.0%) |
| 資料時間跨度 | 2016-2018 年 |

## 專案結構

```
Database/
├── raw/                           # 原始資料檔 (9 個 CSV)
│   ├── olist_customers_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   └── product_category_name_translation.csv
│
├── output_stats/                  # 統計分析輸出 (HTML 圖表與報告)
├── output_decisiontree/           # 決策樹分析輸出
│
├── data_preparation.py            # 資料準備共用模組
├── analysis_correlation.py        # 相關性分析
├── analysis_linear_regression.py  # 線性回歸分析
├── analysis_anova.py              # 變異數分析 (ANOVA)
├── analysis_logistic_regression.py# 邏輯斯回歸分析
├── frequency_analysis.py          # 決策樹分析
├── rmf_recommendation.sas         # RFM 分析與推薦系統 (SAS)
│
├── codebook_by_table.xlsx         # 資料字典
├── rfm_result.xlsx                # RFM 分群結果
└── Olist顧客行為分析報告.md       # 完整分析報告
```

## 分析方法

| 分析方法 | 用途 | 程式檔案 |
|---------|------|---------|
| **相關分析** | 探索變項間關聯強度 | `analysis_correlation.py` |
| **線性回歸** | 預測消費金額的影響因子 | `analysis_linear_regression.py` |
| **ANOVA** | 比較不同群組的消費差異 | `analysis_anova.py` |
| **邏輯斯回歸** | 預測回購機率 | `analysis_logistic_regression.py` |
| **決策樹** | 建立客戶分類規則 | `frequency_analysis.py` |
| **RFM 分析** | 顧客價值分群與推薦 | `rmf_recommendation.sas` |

## 主要發現

### 消費金額影響因素
- **商品數量** (+64 元/件) - 最強正向影響
- **分期期數** (+16 元/期) - 正向影響
- **運費佔比** (-63 元/10%) - 最強負向影響

### 回購行為影響因素
1. **評論分數** - 決策樹特徵重要性達 79.2%
2. **分期期數** - 勝算比 1.07 (p < 0.001)
3. **運費佔比** - 勝算比 1.51

### 回購客 vs 單次客差異

| 指標 | 回購客 | 單次客 | 差異 |
|------|:-----:|:-----:|:---:|
| 平均評分 | 4.19 | 4.14 | +1.3% |
| 提前到貨天數 | 12.59 天 | 11.82 天 | +6.5% |
| 平均分期期數 | 3.2 期 | 2.8 期 | +14% |

## 環境需求

### Python
- Python 3.10+
- 主要套件：
  ```
  pandas
  numpy
  scipy
  statsmodels
  scikit-learn
  plotly
  matplotlib
  seaborn
  ```

### SAS
- SAS 9.4 或更新版本

## 快速開始

### 1. 建立虛擬環境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. 安裝依賴套件
```bash
pip install pandas numpy scipy statsmodels scikit-learn plotly matplotlib seaborn openpyxl
```

### 3. 執行分析
```bash
# 相關性分析
python analysis_correlation.py

# 線性回歸分析
python analysis_linear_regression.py

# ANOVA 分析
python analysis_anova.py

# 邏輯斯回歸分析
python analysis_logistic_regression.py

# 決策樹分析
python frequency_analysis.py
```

### 4. RFM 分析 (SAS)
在 SAS 環境中執行 `rmf_recommendation.sas`

## 輸出結果

- **HTML 互動圖表**：位於 `output_stats/` 和 `output_decisiontree/`
- **分析報告**：`*_report.txt` 文字檔
- **RFM 結果**：`rfm_result.xlsx`

## 資料來源

本專案使用 [Kaggle Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)，由 Olist 提供的真實匿名化電商交易數據。

## 授權

本專案僅供學術研究與教學用途。

---

*Santa Clara University - 資料分析軟體與應用課程專案*
