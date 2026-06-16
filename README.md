# Olist 客戶分析 — 端到端資料分析專案

> 在巴西電商 Olist 的多表交易資料上，用 **SQL 建關聯式資料庫 → RFM / K-means 分群 → cohort 留存 → 複購傾向建模**，端到端回答「客戶是誰、誰會回來」，並誠實揭露模型的資料天花板。

### ▶ 線上互動儀表板：<https://raymond-chen-das.github.io/olist-customer-analysis/>

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-online-C1432B?style=for-the-badge&logo=githubpages&logoColor=white)](https://raymond-chen-das.github.io/olist-customer-analysis/)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-relational%20DB-003B57?logo=sqlite&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-GBDT-EB0F00)
![Plotly](https://img.shields.io/badge/Plotly-viz-3F4F75?logo=plotly&logoColor=white)

---

## 🎯 重點成果
| 發現 | 說明 |
|---|---|
| **97%** 客戶只買一次 | 複購率僅 3% → 這是「獲取型」生意，不是「留存型」（cohort 留存 <0.7%） |
| **16.5%** 高價值客戶 → **31%** 營收 | 高度集中的 80/20，行銷資源應分層投放 |
| 複購主因＝**首單買的品類** | 家用/可補貨 vs 一次性嗜好，複購率差距 ~6× |
| 服務體驗 ≠ 留存 | 配送/評論對複購**不顯著**（p=0.16 / 0.25）；只影響當下滿意度 |
| 複購挖掘模型 lift **1.8×** | 誠實：ROC ~0.59 是「僅憑首單資料」的天花板，非執行問題 |

## 🧠 方法與工程亮點
- **真・SQL 分析**：純 SQL 的 `NTILE` RFM 評分、window function、cohort 留存表（不是用 pandas 補算）。
- **防資料洩漏**：複購模型只用「首單當下」特徵，排除會內含答案的聚合 RFM；前處理只在訓練集 `fit`；並經一次獨立程式碼審查確認 leak-safe。
- **誠實評估**：面對 3% 不平衡，用 PR-AUC / lift / top-decile 而非 Accuracy；壓力測試（5 種子 × 時間切分 × 移除特徵）驗證結論穩健。
- **可重現**：固定隨機種子、鎖定套件版本、由 raw CSV 確定性重建（`00`→`08` 一鍵跑通）。

## 🛠 技術棧
`Python 3.13` · `SQLite` · `pandas` / `numpy` · `scikit-learn` · `XGBoost` · `SHAP` · `SciPy` · `Plotly` · 自製 HTML/CSS/SVG 儀表板

## 🔬 分析流程
| 階段 | 腳本 | 做什麼 | 主要產出 |
|---|---|---|---|
| 資料契約 | `00_validate_contract.py` | 欄位 / 型別 / 缺失 / 主外鍵驗證 | `docs/data-contract.md` |
| 建庫 | `01_build_db.py` | 9 張 CSV → SQLite + `delivered` 視圖 | `data/olist.db` |
| SQL 特徵 | `02_build_features.py` | 客戶特徵表（RFM / 首單）+ cohort 留存 | DB 資料表 |
| EDA | `03_eda.py` | 分布、相關、3 個假設檢定 | `outputs/eda_report.html` |
| 分群 | `04_rfm.py` | RFM 評分箱 + K-means 對照 | `outputs/rfm_segments.html` |
| 複購建模 | `05_propensity.py` | LogReg（防洩漏、排序導向） | `outputs/propensity.html` |
| 模型對照 | `06_xgb_shap.py` | XGBoost + SHAP 驅動因子 | `outputs/xgb_shap.html` |
| 壓力測試 | `07_stress_test.py` | 種子 / 時間 / 移除特徵敏感度 | `docs/evaluation-report.md` |
| 成果報告 | `08_report.py` | 整合成單頁互動儀表板 | `index.html` |

## ⚙️ 環境與執行
```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# 依序執行 00 → 08；精確版本見 requirements.lock.txt
```
> 原始資料（9 張 CSV）請自 Kaggle「Brazilian E-Commerce Public Dataset by Olist」下載後放入 `raw/`。

## 📁 專案結構
- `00`–`08_*.py` — 依序執行的 pipeline
- `sql/` — SQL 腳本（schema / 特徵 / RFM / cohort）
- `src/viz.py` — 共用視覺化工具
- `config.py` — 基準日、複購窗、隨機種子、路徑（集中管理）
- `docs/`、`reports/` — 技術文件與最終報告
- `data/`、`outputs/` — 由腳本重建（不入版控）

## 📌 關鍵方法註記
- 客戶主鍵用 `customer_unique_id`（非每單變動的 `customer_id`）——Olist 最經典的陷阱。
- 複購 base rate ~3% → 視為排序 / 挖掘問題，評估看 PR-AUC / lift / top-decile，而非 Accuracy。
- 時間設計：快照日 `2018-10-17`、複購窗 180 天、首單 ≤ `2018-04-20` 才納入建模（防洩漏）。
