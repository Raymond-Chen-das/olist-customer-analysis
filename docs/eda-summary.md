# EDA 摘要（階段二）

- 資料：`customer_features`（93,358 位客戶，delivered 訂單）；建模集 66,717 位（首單 ≤ 2018-04-20）。
- 視覺化：`outputs/eda_report.html`。
- 原則：假設於規劃階段已定，本階段為驗證；負面結果如實呈現。

## 1. 分布與偏態 → 方法含意
| 特徵 | 觀察 | 含意 |
|---|---|---|
| monetary / avg_order_value / first_order_value | 極右偏 skew≈9（9.6 → 13,664） | 分群前對金額取 log |
| frequency | **97% = 1 單**，skew=11 | **F 維度近乎無變異 → 傳統 RFM 對 Olist 實質退化為「R+M」**；3% 複購客視為特殊小群另處理 |
| recency_days | mean 285、48–761、skew 0.45 | 近似對稱，可直接用 |
| first_review_score | 左偏，mean 4.14、眾數 5 | 多數人給高分 |
| first_delivery_days | 右偏 mean 12、max 209 長尾 | 有離群，建模需 winsorize/log |

## 2. 目標變數
- `repurchase_180d` 正樣本率 **2.96%**（1,976 / 66,717），嚴重不平衡。
- → 評估用 **PR-AUC / lift / top-decile 捕獲率**，不用 Accuracy；建模用 `class_weight`。

## 3. 特徵 → 複購關係（含誠實負面結果）
| 假設 | 檢定 | 結果 | 判定 |
|---|---|---|---|
| H1 首單準時 → 複購 | chi-square | OR=1.13, phi=0.005, p=0.16 | **不顯著（基本無關）** |
| H2 首單評論分 → 複購 | Mann-Whitney U | p=0.25, rank-biserial=−0.014, 中位數皆 5 | **不顯著（基本無關）** |
| H3 配送天數 → 評論分 | Spearman | rho=−0.237, p≈0, n=93,350 | **顯著、效果中等** |

**結論：首單「服務體驗」（配送、評論）不能預測複購；但配送品質確實影響滿意度。**
→ 服務品質是「滿意度槓桿」，不是「留存槓桿」。這個負面結果本身就是洞察。

## 4. 真正帶複購訊號的特徵（補測）
| 特徵 | 訊號 | 用法 |
|---|---|---|
| **首單品類** | 複購率 1.4%–9.0%（baseline 2.96%，~6×）。高：home_appliances 9.0%、fashion_bags 5.1%、bed_bath_table 4.7%(n=6368)、furniture_decor 4.4%；低：cool_stuff 1.4%、electronics 1.7%、consoles 1.9% | **模塊四主訊號**（家用/可補貨 vs 一次性耐久/嗜好） |
| 客戶所在州 | 1.7%（CE/MA）– 3.7%（MT）/ 3.2%（SP, n=26,872） | 中等地理訊號，納入 |
| 首單付款方式 | credit/boleto 2.95%，幾乎無差異 | 不納入 |

## 5. 相關性 / 共線性
- monetary ~ avg_order_value ~ first_order_value：Spearman ρ≈0.98–0.99（因 97% 單筆購買三者趨同）。
- → 模塊三 M 只取 `monetary`；模塊四只取 `first_order_value`（且為非洩漏特徵）。

## 6. 方法決定（EDA → 模塊三/四）
- **模塊三 RFM**：以 R、log(M) 為主軸，F 退化情況記錄在案；K-means 前標準化；k 由輪廓係數＋業務可解釋性決定。
- **模塊四 複購挖掘**：特徵 = 首單品類（分組/編碼）＋ 客戶州 ＋ first_order_value ＋ 首單體驗（預期貢獻低，誠實保留作對照）；LogReg(`class_weight`) 為主、XGBoost 對照；評估 PR-AUC / lift。
- **資料分割**：分層抽樣（維持 2.96%）固定種子為主；時間切分（首單早→晚）作為階段四穩健性檢查。

## 7. 待處理
- 首單 73 個品類多數稀疏 → 需分組（領域分組／或 top-N one-hot + other）；編碼須避免用目標值洩漏。
