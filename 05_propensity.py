"""階段三-四 / 模塊四：複購傾向（高潛力客戶挖掘）。

把複購當成「在 96k 人中撈出 ~3% 高潛力少數」的排序問題：
  - 特徵 = 首單品類 + 客戶州 + first_order_value + 首單體驗（皆首單當下已知，無洩漏）
  - 模型 = LogisticRegression(class_weight='balanced')（可解釋）
  - 評估 = PR-AUC / ROC-AUC / top-decile lift & 捕獲率（非 Accuracy）
  - 解釋 = 係數 → 勝算比；最後 refit 全資料、輸出 customer_scores 排序名單

Run:  ./.venv/Scripts/python.exe 05_propensity.py
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

sys.path.append(str(config.ROOT / "src"))
from viz import save_report  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = config.RANDOM_SEED
NUM = ["log_first_order_value", "first_review_score", "first_delivery_days",
       "first_item_count", "first_freight_ratio"]
BIN = ["first_on_time"]
CAT = ["customer_state", "first_category", "first_order_month"]


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN),
        # min_frequency 自動合併稀疏類別；fit 只在 train，無洩漏
        ("cat", OneHotEncoder(min_frequency=200, handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre),
                     ("lr", LogisticRegression(class_weight="balanced",
                                               max_iter=1000, random_state=SEED))])


def main() -> None:
    con = sqlite3.connect(config.DB_PATH)
    sql = (config.SQL_DIR / "04_model_features.sql").read_text(encoding="utf-8")
    con.execute("DROP TABLE IF EXISTS model_features")
    con.executescript(sql)
    df = pd.read_sql("SELECT * FROM model_features", con)

    df["log_first_order_value"] = np.log1p(df["first_order_value"])
    y = df["repurchase_180d"].astype(int)
    X = df[NUM + BIN + CAT]
    base = y.mean()
    print(f"建模集 {len(df):,}，正樣本率 {base*100:.2f}%\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    pipe = build_pipeline().fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]

    # --- 評估（排序導向）---
    pr_auc = average_precision_score(y_te, proba)
    roc = roc_auc_score(y_te, proba)
    order = np.argsort(-proba)
    ys = y_te.values[order]
    tot_pos = ys.sum()
    top10 = max(1, int(0.10 * len(ys)))
    cap10 = ys[:top10].sum() / tot_pos
    prec10 = ys[:top10].mean()
    lift10 = prec10 / base
    print("[評估 @ 25% 測試集]")
    print(f"  PR-AUC = {pr_auc:.4f}（baseline={base:.4f}，提升 {pr_auc/base:.2f}×）")
    print(f"  ROC-AUC = {roc:.4f}")
    print(f"  Top-decile: 捕獲 {cap10*100:.1f}% 的複購者，"
          f"精準率 {prec10*100:.2f}%（lift {lift10:.2f}×）")

    # --- 可解釋性：係數 → 勝算比 ---
    names = pipe.named_steps["pre"].get_feature_names_out()
    coef = pipe.named_steps["lr"].coef_[0]
    imp = pd.Series(coef, index=[n.split("__")[-1] for n in names]).sort_values()
    top = pd.concat([imp.tail(8), imp.head(6)]).sort_values()
    odds = np.exp(top)
    print("\n[勝算比] 推升/壓低複購機率的特徵 (OR):")
    for name, o in odds.sort_values(ascending=False).items():
        print(f"  {name:32s} OR={o:.2f}")

    # --- HTML ---
    blocks: list = []
    prec, rec, _ = precision_recall_curve(y_te, proba)
    fig_pr = go.Figure(go.Scatter(x=rec, y=prec, mode="lines"))
    fig_pr.add_hline(y=base, line_dash="dash", annotation_text=f"baseline {base:.3f}")
    fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    blocks.append((f"PR 曲線（PR-AUC={pr_auc:.3f}, baseline={base:.3f}）", fig_pr))

    frac = np.arange(1, len(ys) + 1) / len(ys)
    gains = np.cumsum(ys) / tot_pos
    # 持久化 gains 曲線（下採樣 60 點）供成果報告繪 SVG
    gi = np.linspace(0, len(frac) - 1, 60).astype(int)
    pd.DataFrame({"frac": frac[gi], "gains": gains[gi]}).to_sql(
        "model_gains", con, if_exists="replace", index=False)
    fig_g = go.Figure()
    fig_g.add_scatter(x=frac, y=gains, mode="lines", name="模型")
    fig_g.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="隨機",
                      line=dict(dash="dash"))
    fig_g.update_layout(xaxis_title="鎖定客戶比例（依分數高→低）",
                        yaxis_title="累積捕獲複購者比例")
    blocks.append((f"累積捕獲曲線（前 10% 捕獲 {cap10*100:.0f}%）", fig_g))

    fig_c = go.Figure(go.Bar(x=odds.values, y=odds.index, orientation="h"))
    fig_c.add_vline(x=1.0, line_dash="dash")
    fig_c.update_layout(xaxis_title="勝算比 (OR)", yaxis_title="特徵")
    blocks.append(("複購驅動因子（勝算比，>1 推升）", fig_c))
    save_report(blocks, config.OUTPUTS_DIR / "propensity.html",
                "Olist 複購傾向 — 高潛力客戶挖掘")

    # --- refit 全資料、輸出排序名單 ---
    final = build_pipeline().fit(X, y)
    df["repurchase_prob"] = final.predict_proba(X)[:, 1]
    df["high_potential"] = (df["repurchase_prob"].rank(pct=True) >= 0.90).astype(int)
    df[["customer_unique_id", "first_category", "customer_state",
        "repurchase_prob", "high_potential"]].to_sql(
        "customer_scores", con, if_exists="replace", index=False)
    con.commit()
    con.close()
    print(f"\nwrote {config.OUTPUTS_DIR / 'propensity.html'} + customer_scores 表")
    print("\n[experiment-tracking 摘要]")
    print(f"  LogReg(balanced): PR-AUC={pr_auc:.4f}, ROC-AUC={roc:.4f}, "
          f"top-decile 捕獲={cap10*100:.1f}%, lift={lift10:.2f}x")


if __name__ == "__main__":
    main()
