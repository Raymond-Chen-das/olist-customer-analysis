"""階段三 / 模塊四（選配 C）：XGBoost + SHAP 對照。

與 05_propensity.py 同特徵、同切分，公平比較可解釋線性模型 vs 梯度提升樹。
SHAP 用來看非線性驅動因子；輸出 plotly HTML 維持與其他模塊一致。

Run:  ./.venv/Scripts/python.exe 06_xgb_shap.py   （需先跑過 05 以建 model_features）
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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


def top_decile(y_true, proba, base):
    order = np.argsort(-proba)
    ys = y_true.values[order]
    top = max(1, int(0.10 * len(ys)))
    return ys[:top].sum() / ys.sum(), ys[:top].mean() / base


def main() -> None:
    con = sqlite3.connect(config.DB_PATH)
    df = pd.read_sql("SELECT * FROM model_features", con)
    con.close()
    df["log_first_order_value"] = np.log1p(df["first_order_value"])
    y = df["repurchase_180d"].astype(int)
    X = df[NUM + BIN + CAT]
    base = y.mean()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN),
        ("cat", OneHotEncoder(min_frequency=200, handle_unknown="ignore"), CAT),
    ]).fit(X_tr)
    Xtr, Xte = pre.transform(X_tr), pre.transform(X_te)
    names = [n.split("__")[-1] for n in pre.get_feature_names_out()]
    if hasattr(Xtr, "toarray"):
        Xtr, Xte = Xtr.toarray(), Xte.toarray()

    spw = (y_tr == 0).sum() / (y_tr == 1).sum()  # 不平衡權重
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="aucpr", random_state=SEED, n_jobs=4)
    model.fit(Xtr, y_tr)
    proba = model.predict_proba(Xte)[:, 1]

    pr_auc = average_precision_score(y_te, proba)
    roc = roc_auc_score(y_te, proba)
    cap10, lift10 = top_decile(y_te, proba, base)
    print("[XGBoost @ 25% 測試集]")
    print(f"  PR-AUC = {pr_auc:.4f}（baseline={base:.4f}，{pr_auc/base:.2f}×）")
    print(f"  ROC-AUC = {roc:.4f}")
    print(f"  Top-decile 捕獲 {cap10*100:.1f}%，lift {lift10:.2f}×")
    print("\n[對照] LogReg(B): ROC≈0.611, lift≈1.82×  ←→  XGBoost(上)")

    # SHAP：取測試集樣本計算，mean|SHAP| 排序
    samp = shap.sample(Xte, min(3000, len(Xte)), random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(samp)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
    print("\n[SHAP] 平均絕對貢獻 Top 12:")
    for nm, v in mean_abs.head(12).items():
        print(f"  {nm:34s} {v:.4f}")

    blocks: list = []
    bar = mean_abs.head(15).iloc[::-1]
    blocks.append(("SHAP 特徵重要度（平均 |SHAP|，Top 15）",
                   go.Figure(go.Bar(x=bar.values, y=bar.index, orientation="h"))))
    blocks.append(("XGBoost vs LogReg",
                   f"XGBoost：ROC={roc:.3f}、PR-AUC={pr_auc:.3f}、top-decile lift={lift10:.2f}×。"
                   f"LogReg(B)：ROC≈0.611、lift≈1.82×。兩者接近，"
                   "印證訊號天花板由資料決定、非模型選擇；品類仍為主驅動。"))
    save_report(blocks, config.OUTPUTS_DIR / "xgb_shap.html",
                "Olist 複購 — XGBoost + SHAP 對照")
    print(f"\nwrote {config.OUTPUTS_DIR / 'xgb_shap.html'}")
    print(f"\n[experiment-tracking 摘要] XGBoost: PR-AUC={pr_auc:.4f}, "
          f"ROC-AUC={roc:.4f}, top-decile lift={lift10:.2f}x")


if __name__ == "__main__":
    main()
