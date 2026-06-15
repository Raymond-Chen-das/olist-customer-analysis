# Olist 客戶分析 — 端到端作品集

從多表交易資料（巴西電商 Olist）出發，用**關聯式資料庫 + SQL** 做客戶層級行為分析、RFM 分群、cohort 留存，並從新客中**挖掘高複購潛力客戶**。

> 端到端流程：資料契約驗證 → SQL 特徵工程 → 分群 → 複購預測 → 視覺化儀表板，全程可重現、防資料洩漏。

### ▶ 線上互動儀表板：<https://raymond-chen-das.github.io/olist-customer-analysis/>

## 問題與結論
- 97% 客戶只買一次（複購率 3%）；高價值客群 16.5% 客戶貢獻 **31% 營收**。
- 複購由「首單品類」驅動（家用/可補貨 vs 一次性嗜好，~6×）；服務體驗影響的是滿意度、非留存（假設檢定驗證）。
- 高潛力客戶挖掘模型 top-decile lift ~1.8×（誠實：ROC ~0.6 是資料決定的天花板）。
- 完整成果見 `index.html`（視覺儀表板）與 `reports/final-report.md`。

## 環境
```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# exact versions: requirements.lock.txt
```

## 執行順序
```bash
python 00_validate_contract.py  # 階段一：資料契約驗證 -> docs/data-contract.md
python 01_build_db.py           # CSV -> SQLite + delivered 視圖
python 02_build_features.py     # 純 SQL：客戶特徵表 + cohort 留存表
python 03_eda.py                # 階段二 EDA + 假設檢定 -> outputs/eda_report.html
python 04_rfm.py                # 模塊三 RFM 評分箱(NTILE) + K-means -> outputs/rfm_segments.html
python 05_propensity.py         # 模塊四 複購挖掘 LogReg -> outputs/propensity.html
python 06_xgb_shap.py           # 選配：XGBoost + SHAP 對照
python 07_stress_test.py        # 階段四 壓力測試 -> docs/evaluation-report.md
python 08_report.py             # 模塊五 成果儀表板 -> index.html（根目錄，供 GitHub Pages）
```

## 結構
- `config.py` — 基準日、複購窗、隨機種子、路徑（集中管理）
- `sql/` — SQL 腳本（schema / 特徵 / RFM / cohort）
- `src/` — 可重用函式
- `data/olist.db` — SQLite（不入版控）
- `outputs/` — HTML 圖表與特徵表
- `docs/`、`reports/` — 流程文件與最終報告

## 關鍵方法註記
- 客戶主鍵用 `customer_unique_id`（非每單變動的 `customer_id`）。
- 複購建模 base rate ~3.12% → 視為排序/挖掘問題，評估看 PR-AUC / lift / top-decile，非 Accuracy。
- 時間設計：快照日 2018-10-17、複購窗 180 天、首單 ≤ 2018-04-20 才納入建模（防洩漏）。
