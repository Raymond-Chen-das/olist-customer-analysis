"""階段二：EDA — 分布/偏態、目標不平衡、特徵→目標關係、相關性、假設檢定。

讀 customer_features，產出 outputs/eda_report.html 與 console 摘要
（供撰寫 docs/eda-summary.md）。研究設計先行：假設於規劃階段已定
（配送/評論/複購之關係），此處為驗證而非事後挑選。

Run:  ./.venv/Scripts/python.exe 03_eda.py
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats

import config

sys.path.append(str(config.ROOT / "src"))
from viz import save_report  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NUM_COLS = ["frequency", "monetary", "avg_order_value", "recency_days",
            "first_order_value", "first_review_score", "first_delivery_days"]


def main() -> None:
    con = sqlite3.connect(config.DB_PATH)
    df = pd.read_sql("SELECT * FROM customer_features", con)
    con.close()
    n = len(df)
    blocks: list = []
    print("=" * 60, "\n階段二 EDA 摘要\n", "=" * 60, sep="")
    print(f"客戶數: {n:,}")

    # ---- A. 分布與偏態 ----
    desc = df[NUM_COLS].describe().T
    desc["skew"] = df[NUM_COLS].skew()
    print("\n[A] 數值特徵分布:")
    print(desc[["mean", "50%", "std", "min", "max", "skew"]].round(2).to_string())

    fc = df["frequency"].clip(upper=4).value_counts().sort_index()
    blocks.append(("購買頻率分布（97% 只買一次）",
                   px.bar(x=fc.index.astype(str), y=fc.values,
                          labels={"x": "delivered 訂單數 (4=4+)", "y": "客戶數"})))
    blocks.append(("Monetary 分布（右偏，y 軸 log）",
                   px.histogram(df, x="monetary", nbins=60, log_y=True,
                                labels={"monetary": "客戶總金額 (BRL)"})))
    blocks.append(("Recency 分布",
                   px.histogram(df, x="recency_days", nbins=60,
                                labels={"recency_days": "距快照日天數"})))
    rs = df["first_review_score"].round().value_counts().sort_index()
    blocks.append(("首單評論分數分布",
                   px.bar(x=rs.index.astype(str), y=rs.values,
                          labels={"x": "評論分數", "y": "客戶數"})))
    blocks.append(("首單配送天數分布",
                   px.histogram(df.query("first_delivery_days>=0"),
                                x="first_delivery_days", nbins=60,
                                labels={"first_delivery_days": "配送天數"})))

    # ---- B. 目標變數 ----
    model = df[df["repurchase_180d"].notna()].copy()
    model["repurchase_180d"] = model["repurchase_180d"].astype(int)
    pos = model["repurchase_180d"].mean()
    print(f"\n[B] 目標 repurchase_180d：建模集 {len(model):,}，"
          f"正樣本率 {pos*100:.2f}%（嚴重不平衡）")
    blocks.append(("目標變數：180 天複購率",
                   f"建模集 {len(model):,} 位（首單 ≤ {config.MODEL_FIRST_ORDER_CUTOFF.date()}），"
                   f"正樣本 {int(model['repurchase_180d'].sum()):,}（{pos*100:.2f}%）。"
                   "嚴重不平衡 → 評估用 PR-AUC / lift，非 Accuracy。"))

    # ---- C. 特徵 → 目標關係 ----
    print("\n[C] 首單特徵 vs 複購率:")
    m_ot = model.dropna(subset=["first_on_time"]).copy()
    m_ot["first_on_time"] = m_ot["first_on_time"].astype(int)
    ot = m_ot.groupby("first_on_time")["repurchase_180d"].agg(["mean", "size"])
    print("  首單準時(1) vs 延遲(0):")
    print(ot.assign(rate_pct=(ot["mean"] * 100).round(2)).to_string())
    blocks.append(("複購率：首單準時 vs 延遲",
                   px.bar(x=["延遲 (0)", "準時 (1)"], y=(ot["mean"] * 100).values,
                          labels={"x": "首單是否準時", "y": "複購率 %"})))

    rv = model.groupby(model["first_review_score"].round())["repurchase_180d"].agg(["mean", "size"])
    print("  依首單評論分:")
    print(rv.assign(rate_pct=(rv["mean"] * 100).round(2)).to_string())
    blocks.append(("複購率 vs 首單評論分數",
                   px.line(x=rv.index.astype(str), y=(rv["mean"] * 100).values, markers=True,
                           labels={"x": "首單評論分數", "y": "複購率 %"})))

    # ---- D. 相關性 / 共線性 ----
    corr = df[NUM_COLS + ["first_on_time"]].corr(method="spearman")
    print("\n[D] Spearman 相關 — |r|>0.5 視為高共線:")
    cols = corr.columns.tolist()
    high = [(cols[i], cols[j], corr.iloc[i, j])
            for i in range(len(cols)) for j in range(i + 1, len(cols))
            if abs(corr.iloc[i, j]) > 0.5]
    for a, b, r in high:
        print(f"  [!] {a} ~ {b}: {r:.2f}")
    if not high:
        print("  （無 |r|>0.5 的配對）")
    blocks.append(("特徵 Spearman 相關熱圖",
                   px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu",
                             zmin=-1, zmax=1, aspect="auto")))

    # ---- E. 各州複購率（地理）----
    st = (model.groupby("customer_state")["repurchase_180d"]
          .agg(["mean", "size"]).query("size>=500").sort_values("mean", ascending=False))
    print("\n[E] 各州複購率 (建模集, n>=500):")
    print(st.assign(rate_pct=(st["mean"] * 100).round(2)).to_string())
    blocks.append(("各州複購率（n≥500）",
                   px.bar(x=st.index, y=(st["mean"] * 100).values,
                          labels={"x": "客戶所在州", "y": "複購率 %"})))

    # ---- F. 假設檢定（含效果量）----
    print("\n[F] 假設檢定:")
    ct = pd.crosstab(m_ot["first_on_time"], m_ot["repurchase_180d"])
    chi2, p1, _, _ = stats.chi2_contingency(ct)
    phi = np.sqrt(chi2 / ct.values.sum())
    a, b = ct.loc[1, 1], ct.loc[1, 0]
    c, d = ct.loc[0, 1], ct.loc[0, 0]
    odds = (a * d) / (b * c)
    print(f"  H1 首單準時→複購 (chi-square): chi2={chi2:.1f}, p={p1:.2e}, "
          f"phi={phi:.3f}, OR(準時/延遲)={odds:.2f}")

    g1 = model.loc[model.repurchase_180d == 1, "first_review_score"]
    g0 = model.loc[model.repurchase_180d == 0, "first_review_score"]
    u, p2 = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    rbc = 1 - 2 * u / (len(g1) * len(g0))  # rank-biserial effect size
    print(f"  H2 首單評論分 vs 複購 (Mann-Whitney U): p={p2:.2e}, "
          f"rank-biserial={rbc:.3f}, 中位數 複購={g1.median():.0f}/未複購={g0.median():.0f}")

    sub = df.query("first_delivery_days>=0").dropna(subset=["first_delivery_days", "first_review_score"])
    rho, p3 = stats.spearmanr(sub["first_delivery_days"], sub["first_review_score"])
    print(f"  H3 配送天數 vs 評論分 (Spearman): rho={rho:.3f}, p={p3:.2e}, n={len(sub):,}")

    blocks.append(("假設檢定摘要",
                   f"H1 首單準時 vs 延遲的複購勝算比 OR={odds:.2f}（phi={phi:.3f}, p={p1:.1e}）；"
                   f"H2 評論分對複購 rank-biserial={rbc:.3f}（p={p2:.1e}）；"
                   f"H3 配送越久評論越低 Spearman rho={rho:.3f}（p={p3:.1e}）。"))

    out = config.OUTPUTS_DIR / "eda_report.html"
    save_report(blocks, out, "Olist EDA — 客戶行為與複購驅動因子")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
