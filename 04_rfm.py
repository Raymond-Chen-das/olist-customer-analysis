"""階段三 / 模塊三：RFM 分群（SQL NTILE 評分 + K-means 對照）。

流程：
  1. 執行 sql/03_rfm_scores.sql 產生 rfm_scores（傳統評分箱 → 命名客群）。
  2. K-means 對照：以 R + log(M) 分群（F 退化故排除），輪廓係數選 k。
  3. 交叉比對兩種分群，寫 customer_segments 表，產 HTML 與實驗紀錄。

Run:  ./.venv/Scripts/python.exe 04_rfm.py
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config

sys.path.append(str(config.ROOT / "src"))
from viz import save_report  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = config.RANDOM_SEED


def main() -> None:
    con = sqlite3.connect(config.DB_PATH)

    # 1. SQL 傳統 RFM 評分 ---------------------------------------------------
    sql = (config.SQL_DIR / "03_rfm_scores.sql").read_text(encoding="utf-8")
    con.execute("DROP TABLE IF EXISTS rfm_scores")
    con.executescript(sql)
    df = pd.read_sql("SELECT * FROM rfm_scores", con)
    n = len(df)
    print(f"rfm_scores: {n:,} 位客戶\n")

    seg = df["rfm_segment"].value_counts()
    print("[RFM 評分箱] 客群分布:")
    for name, cnt in seg.items():
        print(f"  {name:24s} {cnt:>7,} ({cnt/n*100:5.2f}%)")

    # 2. K-means 對照（R + log(M)，F 退化排除）------------------------------
    df["log_monetary"] = np.log1p(df["monetary"])
    X = StandardScaler().fit_transform(df[["recency_days", "log_monetary"]])

    print("\n[K-means] 輪廓係數掃描 (sample=10000):")
    sil = {}
    for k in config.KMEANS_K_CANDIDATES:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        sil[k] = silhouette_score(X, km.labels_, sample_size=10000, random_state=SEED)
        print(f"  k={k}: silhouette={sil[k]:.4f}")
    # 選 k：商業可用區間 [3,6] 內輪廓最高（避免 k=2 的瑣碎二分）
    best_k = max((k for k in sil if 3 <= k <= 6), key=lambda k: sil[k])
    print(f"  -> 選 k={best_k}（[3,6] 區間內輪廓最高；兼顧業務可解釋性）")

    km = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit(X)
    df["kmeans_cluster"] = km.labels_

    print(f"\n[K-means] k={best_k} 各群輪廓:")
    prof = df.groupby("kmeans_cluster").agg(
        size=("customer_unique_id", "size"),
        recency=("recency_days", "mean"),
        monetary=("monetary", "mean"),
        frequency=("frequency", "mean")).round(1)
    print(prof.to_string())

    # 3. 兩種分群交叉比對 ----------------------------------------------------
    print("\n[對照] RFM 評分箱 vs K-means 群（列：RFM 段，欄：cluster）:")
    ct = pd.crosstab(df["rfm_segment"], df["kmeans_cluster"])
    print(ct.to_string())

    # 4. 寫 customer_segments 表 --------------------------------------------
    out_cols = ["customer_unique_id", "recency_days", "frequency", "monetary",
                "r_score", "f_score", "m_score", "rfm_segment", "kmeans_cluster"]
    df[out_cols].to_sql("customer_segments", con, if_exists="replace", index=False)
    con.commit()
    con.close()

    # 5. HTML --------------------------------------------------------------
    blocks: list = []
    blocks.append(("RFM 評分箱客群分布",
                   px.bar(x=seg.values, y=seg.index, orientation="h",
                          labels={"x": "客戶數", "y": "客群"})))
    blocks.append(("K-means 輪廓係數 vs k",
                   px.line(x=list(sil.keys()), y=list(sil.values()), markers=True,
                           labels={"x": "k", "y": "silhouette"})))
    samp = df.sample(n=min(8000, n), random_state=SEED)
    blocks.append(("客戶分布：Recency vs log(Monetary)（顏色＝K-means 群）",
                   px.scatter(samp, x="recency_days", y="log_monetary",
                              color=samp["kmeans_cluster"].astype(str),
                              opacity=0.5, labels={"color": "cluster"})))
    blocks.append((f"K-means 各群平均（k={best_k}）",
                   px.bar(prof.reset_index(), x="kmeans_cluster", y="monetary",
                          labels={"monetary": "平均金額 (BRL)", "kmeans_cluster": "群"})))
    save_report(blocks, config.OUTPUTS_DIR / "rfm_segments.html",
                "Olist 客戶分群 — RFM 評分箱 × K-means")
    print(f"\nwrote {config.OUTPUTS_DIR / 'rfm_segments.html'}")

    # 給實驗紀錄用的摘要
    print("\n[experiment-tracking 摘要]")
    print(f"  silhouette: " + ", ".join(f"k{k}={v:.3f}" for k, v in sil.items()))
    print(f"  chosen k={best_k}")


if __name__ == "__main__":
    main()
