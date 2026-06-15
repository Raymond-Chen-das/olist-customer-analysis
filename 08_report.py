"""模塊五 / 階段五：成果報告（editorial data dashboard）。

從 DB 取真實數字，產出單頁互動感儀表板 reports/final_report.html（+ 根目錄 index.html
供 GitHub Pages 直接連結）。包含：KPI 帶、cohort 留存熱圖、獲取成長 sparkbar、
營收集中度、SVG gains 曲線、客群表、品類驅動。細部互動圖見 outputs/*.html。

Run:  ./.venv/Scripts/python.exe 08_report.py
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pandas as pd

import config

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEG_ORDER = ["高價值 (Champions)", "忠誠潛力 (Loyal)", "高額流失風險 (At Risk)",
             "新客待培養 (New)", "一般 (Regular)", "沉睡 (Hibernating)"]
ACTIONS = {
    "高價值 (Champions)": "VIP 維繫、新品優先、推薦獎勵",
    "忠誠潛力 (Loyal)": "升級／加購誘因、交叉銷售",
    "高額流失風險 (At Risk)": "限時召回、主動滿意度關懷",
    "新客待培養 (New)": "第二單誘因、確保首單配送",
    "一般 (Regular)": "輕度經營、節慶觸發",
    "沉睡 (Hibernating)": "低成本喚醒，反應差則放生",
}

CSS = """
:root{--paper:#F3EFE6;--paper-2:#ECE5D6;--ink:#1C1814;--ink-soft:#5A5147;
--rule:rgba(28,24,20,.16);--accent:#C1432B;--accent-2:#1F4D43;--gold:#A8782F}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--paper);color:var(--ink);font-family:"Fraunces","Noto Serif TC",Georgia,serif;
font-optical-sizing:auto;line-height:1.6;-webkit-font-smoothing:antialiased;position:relative;overflow-x:hidden}
body::before{content:"";position:fixed;inset:0;z-index:9999;pointer-events:none;opacity:.5;mix-blend-mode:multiply;
background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.05'/%3E%3C/svg%3E")}
.wrap{max-width:1080px;margin:0 auto;padding:0 32px}
.mono{font-family:"IBM Plex Mono","Noto Sans TC",monospace}
.eyebrow{font-family:"IBM Plex Mono","Noto Sans TC",monospace;font-size:.72rem;letter-spacing:.28em;
text-transform:uppercase;color:var(--accent);font-weight:500}

/* nav */
.nav{position:sticky;top:0;z-index:50;background:rgba(243,239,230,.86);backdrop-filter:blur(9px);border-bottom:1px solid var(--rule)}
.nav .wrap{display:flex;align-items:center;justify-content:space-between;height:54px}
.nav .brand{font-family:"IBM Plex Mono",monospace;font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;font-weight:600}
.nav .links{display:flex;gap:24px}
.nav .links a{font-family:"IBM Plex Mono",monospace;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-soft);text-decoration:none;transition:color .2s}
.nav .links a:hover{color:var(--accent)}
@media(max-width:720px){.nav .links{display:none}}

/* masthead */
.masthead{padding:80px 0 42px;border-bottom:1.5px solid var(--ink)}
.masthead .eyebrow{margin-bottom:26px}
h1{font-weight:900;font-size:clamp(2.8rem,8.5vw,6rem);line-height:.94;letter-spacing:-.02em;margin-bottom:24px}
h1 em{font-style:italic;font-weight:500;color:var(--accent)}
.dek{font-size:clamp(1.05rem,2vw,1.35rem);max-width:48ch;color:var(--ink-soft)}
.meta{margin-top:32px;display:flex;gap:26px;flex-wrap:wrap;font-family:"IBM Plex Mono","Noto Sans TC",monospace;
font-size:.74rem;letter-spacing:.07em;color:var(--ink-soft);text-transform:uppercase}
.meta b{color:var(--ink);font-weight:600}

/* stats */
.stats{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1.5px solid var(--ink)}
.stat{padding:38px 24px 36px;border-right:1px solid var(--rule)}
.stat:last-child{border-right:none}
.stat .num{font-size:clamp(2.4rem,5vw,3.8rem);font-weight:900;line-height:1;letter-spacing:-.03em}
.stat:nth-child(odd) .num{color:var(--accent)}
.stat .cap{margin-top:13px;font-family:"IBM Plex Mono","Noto Sans TC",monospace;font-size:.73rem;letter-spacing:.03em;color:var(--ink-soft);line-height:1.5}
@media(max-width:720px){.stats{grid-template-columns:repeat(2,1fr)}.stat:nth-child(2n){border-right:none}}

/* sections */
section.blk{padding:70px 0;border-bottom:1px solid var(--rule)}
.sec-head{display:flex;align-items:baseline;gap:18px;margin-bottom:30px}
.sec-no{font-family:"IBM Plex Mono",monospace;font-size:.9rem;color:var(--accent);font-weight:600}
h2{font-size:clamp(1.6rem,3.3vw,2.5rem);font-weight:600;letter-spacing:-.01em;line-height:1.12}
.lede{font-size:1.12rem;max-width:64ch;color:var(--ink-soft);margin-bottom:6px}
.lede b{color:var(--ink)}

/* findings */
.finds{display:grid;grid-template-columns:repeat(3,1fr);margin-top:8px}
.find{padding:28px 24px 26px 0;border-top:2px solid var(--ink)}
.find:not(:last-child){border-right:1px solid var(--rule);padding-right:24px}
.find .fnum{font-family:"IBM Plex Mono",monospace;color:var(--accent);font-size:.78rem;letter-spacing:.1em}
.find h3{font-size:1.28rem;font-weight:600;margin:13px 0 9px;line-height:1.25}
.find p{font-size:.96rem;color:var(--ink-soft)}
@media(max-width:720px){.finds{grid-template-columns:1fr}.find{border-right:none!important;padding-right:0}}

/* bars */
.bars{margin-top:8px}
.bar-row{display:grid;grid-template-columns:200px 1fr auto;align-items:center;gap:18px;padding:9px 0;border-bottom:1px solid var(--rule)}
.bar-label{font-size:.97rem}
.bar-cat{font-family:"IBM Plex Mono",monospace;font-size:.85rem}
.bar-track{height:14px;background:rgba(28,24,20,.07);position:relative;overflow:hidden}
.bar-fill{position:absolute;inset:0 auto 0 0;width:0;animation:grow 1.1s cubic-bezier(.2,.8,.2,1) forwards;animation-delay:var(--d,.3s)}
.bar-val{font-family:"IBM Plex Mono",monospace;font-size:.9rem;font-weight:500;min-width:60px;text-align:right}
@keyframes grow{to{width:var(--w)}}
@media(max-width:620px){.bar-row{grid-template-columns:128px 1fr auto;gap:10px}.bar-label,.bar-cat{font-size:.8rem}}

/* cohort heatmap */
.split{display:grid;grid-template-columns:1.5fr 1fr;gap:40px;align-items:start}
@media(max-width:820px){.split{grid-template-columns:1fr;gap:30px}}
.heat{overflow-x:auto}
.heat-grid{display:grid;gap:3px;min-width:430px}
.heat-h{font-family:"IBM Plex Mono",monospace;font-size:.64rem;color:var(--ink-soft);text-align:center;padding-bottom:4px}
.heat-row-label{font-family:"IBM Plex Mono",monospace;font-size:.66rem;color:var(--ink-soft);display:flex;align-items:center}
.heat-cell{aspect-ratio:1/.6;display:flex;align-items:center;justify-content:center;font-family:"IBM Plex Mono",monospace;font-size:.6rem;color:#fff;border-radius:1px}
.subcap{font-family:"IBM Plex Mono",monospace;font-size:.68rem;color:var(--ink-soft);margin:8px 0 2px;letter-spacing:.03em}
.spark{display:flex;align-items:flex-end;gap:3px;height:120px;margin-top:6px;border-bottom:1px solid var(--rule)}
.spark .b{flex:1;background:var(--accent-2);opacity:.82;transition:opacity .2s}
.spark .b:hover{opacity:1}

/* concentration */
.conc-row{margin:18px 0}
.conc-row .cl{display:flex;justify-content:space-between;align-items:baseline;font-family:"IBM Plex Mono",monospace;font-size:.82rem;margin-bottom:7px}
.conc-row .cl b{font-size:1.1rem;color:var(--ink)}
.conc-track{height:30px;background:rgba(28,24,20,.07);position:relative;overflow:hidden}
.conc-fill{height:100%;width:0;background:var(--accent);animation:grow 1.2s cubic-bezier(.2,.8,.2,1) forwards}

/* table */
.tbl{width:100%;border-collapse:collapse;margin-top:20px;font-size:.95rem}
.tbl th{font-family:"IBM Plex Mono","Noto Sans TC",monospace;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;
text-align:left;padding:12px 13px;color:var(--paper);background:var(--ink);font-weight:500}
.tbl td{padding:12px 13px;border-bottom:1px solid var(--rule)}
.tbl td.n{font-family:"IBM Plex Mono",monospace;text-align:right}
.tbl tbody tr{transition:background .18s}
.tbl tbody tr:hover{background:var(--paper-2)}
.tbl tbody tr:hover td:first-child{box-shadow:inset 3px 0 0 var(--accent)}

/* gains svg */
.gains svg{width:100%;height:auto;display:block;margin-top:6px}
.gl{stroke:rgba(28,24,20,.1)}
.diag{stroke:var(--ink-soft);stroke-dasharray:5 5;stroke-width:1.1}
.curve{fill:none;stroke:var(--accent);stroke-width:2.6;stroke-linejoin:round}
.mk{stroke:var(--accent-2);stroke-width:1.4;stroke-dasharray:3 3}
.dot{fill:var(--accent-2)}
.lab,.ax{font-family:"IBM Plex Mono","Noto Sans TC",monospace;fill:var(--ink-soft)}
.lab{font-size:13px;fill:var(--ink)}.ax{font-size:11px}

/* model cards + aside */
.model-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin:8px 0 4px}
.mcard{border-top:2px solid var(--ink);padding-top:15px}
.mcard .mv{font-size:2.1rem;font-weight:900;letter-spacing:-.02em}
.mcard .ml{font-family:"IBM Plex Mono",monospace;font-size:.72rem;color:var(--ink-soft);margin-top:6px}
@media(max-width:720px){.model-grid{grid-template-columns:1fr}}
.aside{margin-top:30px;padding:24px 26px;background:var(--paper-2);border-left:3px solid var(--accent)}
.aside .tag{font-family:"IBM Plex Mono",monospace;font-size:.7rem;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);display:block;margin-bottom:11px}
.aside p{color:var(--ink-soft);font-size:.96rem}.aside p+p{margin-top:9px}.aside b{color:var(--ink)}

footer{padding:52px 0 78px}
footer .eyebrow{margin-bottom:16px}
footer p{color:var(--ink-soft);font-size:.9rem;max-width:66ch}
footer .files{font-family:"IBM Plex Mono",monospace;font-size:.78rem;margin-top:14px;color:var(--ink-soft);line-height:2}

.reveal{opacity:0;transform:translateY(18px);animation:fadeUp .85s cubic-bezier(.2,.7,.2,1) forwards;animation-delay:var(--rd,0ms)}
@keyframes fadeUp{to{opacity:1;transform:none}}
@media(prefers-reduced-motion:reduce){.reveal{animation:none;opacity:1;transform:none}.bar-fill,.conc-fill{animation:none;width:var(--w,100%)}}
"""


def bars(items):
    out = ['<div class="bars">']
    for i, (label, w, disp, color, cls) in enumerate(items):
        out.append(f'<div class="bar-row" style="--d:{0.25+i*0.06:.2f}s">'
                   f'<span class="bar-label {cls}">{label}</span>'
                   f'<span class="bar-track"><span class="bar-fill" style="--w:{w:.1f}%;background:var({color})"></span></span>'
                   f'<span class="bar-val">{disp}</span></div>')
    return "".join(out) + '</div>'


def heatmap(piv, maxret):
    offs = [c for c in range(1, 7) if c in piv.columns]
    cells = [f'<div class="heat-grid" style="grid-template-columns:52px repeat({len(offs)},1fr)">',
             '<div></div>'] + [f'<div class="heat-h">+{m}</div>' for m in offs]
    for coh, row in piv.iterrows():
        cells.append(f'<div class="heat-row-label">{coh}</div>')
        for m in offs:
            v = float(row.get(m, 0) or 0)
            a = 0.08 + min(1.0, v / maxret) * 0.92 if maxret else 0.08
            txt = f"{v:.1f}" if v >= 0.4 else ""
            cells.append(f'<div class="heat-cell" style="background:rgba(193,67,43,{a:.2f})">{txt}</div>')
    return "".join(cells) + '</div>'


def spark(items, maxv):
    out = ['<div class="spark">']
    for coh, sz in items:
        out.append(f'<div class="b" style="height:{sz/maxv*100:.0f}%" title="{coh}: {sz:,}"></div>')
    return "".join(out) + '</div>'


def conc(rows):
    out = []
    for i, (label, pct, val) in enumerate(rows):
        out.append(f'<div class="conc-row"><div class="cl"><span>{label}</span><b>{val}</b></div>'
                   f'<div class="conc-track"><div class="conc-fill" style="--w:{pct:.1f}%;width:{pct:.1f}%;animation-delay:{0.3+i*0.18:.2f}s"></div></div></div>')
    return "".join(out)


def gains_svg(frac, gains):
    W, H, pad = 620, 280, 42
    X = lambda f: pad + f * (W - 2 * pad)
    Y = lambda g: (H - pad) - g * (H - 2 * pad)
    pts = " ".join(f"{X(f):.1f},{Y(g):.1f}" for f, g in zip(frac, gains))
    area = f"{X(0):.1f},{Y(0):.1f} " + pts + f" {X(1):.1f},{Y(0):.1f}"
    dg = float(np.interp(0.1, frac, gains))
    grid = "".join(f'<line x1="{pad}" y1="{Y(t):.1f}" x2="{W-pad}" y2="{Y(t):.1f}" class="gl"/>'
                   for t in [.25, .5, .75, 1])
    return (f'<div class="gains"><svg viewBox="0 0 {W} {H}" role="img" aria-label="累積捕獲曲線">'
            f'{grid}<polygon points="{area}" fill="rgba(193,67,43,.10)"/>'
            f'<line x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(1):.1f}" y2="{Y(1):.1f}" class="diag"/>'
            f'<polyline points="{pts}" class="curve"/>'
            f'<line x1="{X(.1):.1f}" y1="{Y(0):.1f}" x2="{X(.1):.1f}" y2="{Y(dg):.1f}" class="mk"/>'
            f'<circle cx="{X(.1):.1f}" cy="{Y(dg):.1f}" r="4.5" class="dot"/>'
            f'<text x="{X(.1)+9:.1f}" y="{Y(dg)-9:.1f}" class="lab">前 10% 客戶 → 捕獲 {dg*100:.0f}% 複購者</text>'
            f'<text x="{pad}" y="{H-14}" class="ax">0%</text>'
            f'<text x="{W-pad-58:.1f}" y="{H-14}" class="ax">100% 客戶（依分數）</text>'
            f'<text x="{pad-6}" y="{Y(1)+4:.1f}" class="ax" text-anchor="end">100%</text>'
            '</svg></div>')


def main() -> None:
    con = sqlite3.connect(config.DB_PATH)
    seg = pd.read_sql("SELECT * FROM customer_segments", con)
    mf = pd.read_sql("SELECT first_category, repurchase_180d FROM model_features", con)
    cr = pd.read_sql("SELECT * FROM cohort_retention", con)
    gains_df = pd.read_sql("SELECT * FROM model_gains ORDER BY frac", con)
    con.close()

    n = len(seg)
    repeat = (seg["frequency"] >= 2).mean()
    champ = seg[seg["rfm_segment"].str.startswith("高價值")]
    champ_share = champ["monetary"].sum() / seg["monetary"].sum()
    champ_pct = len(champ) / n
    base = mf["repurchase_180d"].mean()

    # frequency bars
    fd = seg["frequency"].clip(upper=4).value_counts().sort_index()
    flab = {1: "1 單（單次買家）", 2: "2 單", 3: "3 單", 4: "4 單以上"}
    fmax = fd.max()
    fbars = bars([(flab[k], fd[k]/fmax*100, f"{fd[k]:,}", "--accent" if k == 1 else "--accent-2", "")
                  for k in fd.index])

    # cohort
    crf = cr[(cr.cohort >= "2017-01") & (cr.cohort <= "2018-06") & (cr.cohort_size >= 300)]
    piv = crf.pivot(index="cohort", columns="month_offset", values="retention_pct")
    off = piv[[c for c in range(1, 7) if c in piv.columns]]
    maxret = float(off.max().max())
    heat = heatmap(piv, maxret)
    szl = crf[crf.month_offset == 0].sort_values("cohort")[["cohort", "cohort_size"]].values.tolist()
    sp = spark(szl, max(s for _, s in szl))

    # segments
    sizes = seg["rfm_segment"].value_counts()
    smax = sizes.max()
    order = [s for s in SEG_ORDER if s in sizes.index]
    sbars = bars([(s, sizes[s]/smax*100, f"{sizes[s]/n*100:.1f}%", "--accent", "") for s in order])
    prof = seg.groupby("rfm_segment").agg(size=("customer_unique_id", "size"),
                                          monetary=("monetary", "mean"), recency=("recency_days", "mean"))
    trows = "".join(f"<tr><td>{s}</td><td class='n'>{int(prof.loc[s,'size']):,}</td>"
                    f"<td class='n'>{prof.loc[s,'monetary']:.0f}</td>"
                    f"<td class='n'>{prof.loc[s,'recency']:.0f}</td><td>{ACTIONS.get(s,'')}</td></tr>"
                    for s in order)
    conc_html = conc([("高價值客戶佔總客戶數", champ_pct*100, f"{champ_pct*100:.1f}%"),
                      ("其貢獻的營收佔比", champ_share*100, f"{champ_share*100:.0f}%")])

    # category
    cat = (mf.groupby("first_category")["repurchase_180d"].agg(["mean", "size"])
           .query("size>=400").sort_values("mean", ascending=False))
    pick = pd.concat([cat.head(6), cat.tail(6)])
    cmax = pick["mean"].max()*100
    cbars = bars([(idx, r["mean"]*100/cmax*100, f"{r['mean']*100:.1f}%",
                   "--accent" if r["mean"] >= base else "--accent-2", "bar-cat")
                  for idx, r in pick.iterrows()])

    gsvg = gains_svg(gains_df["frac"].values, gains_df["gains"].values)

    body = f"""
<nav class="nav"><div class="wrap">
  <span class="brand">Olist · Customer Analytics</span>
  <span class="links"><a href="#find">發現</a><a href="#cohort">留存</a><a href="#seg">客群</a><a href="#drv">複購驅動</a><a href="#mdl">模型</a></span>
</div></nav>
<div class="wrap">
  <header class="masthead reveal">
    <div class="eyebrow">Customer Analytics Dossier · Olist E-commerce</div>
    <h1>回購的人，<br><em>買了什麼？</em></h1>
    <p class="dek">97% 的客戶只來一次。本案用 SQL、RFM 與 cohort 留存分析，找出關鍵的 3%——並用假設檢定推翻了一個你以為理所當然的假設。</p>
    <div class="meta"><span><b>資料</b> Olist 2016–2018</span><span><b>客戶</b> {n:,} 位</span>
      <span><b>方法</b> SQL · RFM · Cohort · LogReg/XGBoost</span><span><b>作者</b> raymond.chen.das</span></div>
  </header>

  <section class="stats reveal" style="--rd:100ms">
    <div class="stat"><div class="num">97%</div><div class="cap">客戶只買一次<br>單次交易為主</div></div>
    <div class="stat"><div class="num">{repeat*100:.1f}%</div><div class="cap">整體複購率<br>留存是難題</div></div>
    <div class="stat"><div class="num">{champ_share*100:.0f}%</div><div class="cap">營收來自前 {champ_pct*100:.0f}%<br>高價值客戶</div></div>
    <div class="stat"><div class="num">1.8×</div><div class="cap">高潛力名單<br>top-decile lift</div></div>
  </section>

  <section id="find" class="blk reveal">
    <div class="sec-head"><span class="sec-no">01</span><h2>三個主要發現</h2></div>
    <div class="finds">
      <div class="find"><div class="fnum">FINDING 01</div><h3>九成七只來一次</h3>
        <p>頻率維度幾乎無變異，傳統 RFM 實質退化為「近期 × 金額」。留存難，獲取品質才是主戰場。</p></div>
      <div class="find"><div class="fnum">FINDING 02</div><h3>16.5% 客戶 = 31% 營收</h3>
        <p>高價值客群高度集中，典型 80/20。行銷資源應依客群分層投放，而非齊頭式。</p></div>
      <div class="find"><div class="fnum">FINDING 03</div><h3>留存看「買什麼」</h3>
        <p>複購由首單品類驅動（家用 vs 嗜好約 6×）。配送與評論影響的是<b>滿意度、不是留存</b>——假設檢定 p=0.16 / 0.25 不顯著。</p></div>
    </div>
  </section>

  <section id="cohort" class="blk reveal">
    <div class="sec-head"><span class="sec-no">02</span><h2>留存懸崖：一個獲取型、而非留存型的生意</h2></div>
    <p class="lede">純 SQL cohort 分析。第 0 月（首購當月）必為 100%，<b>其後每一個月留存都崩跌到 1% 以下</b>——這就是為什麼留存不是這個生意的槓桿。</p>
    <div class="split">
      <div>
        <div class="subcap">▸ 第 1–6 月留存率（%，自有色階，最亮僅 {maxret:.2f}%）</div>
        <div class="heat">{heat}</div>
      </div>
      <div>
        <div class="subcap">▸ 每月新客 cohort 規模（獲取成長）</div>
        {sp}
        <p class="lede" style="font-size:.95rem;margin-top:14px">獲取端強勁成長（月新客 717 → 7,000+）；但留存幾近於零 → 成長完全靠「拉新」，一旦獲取放緩即見頂。</p>
      </div>
    </div>
  </section>

  <section id="seg" class="blk reveal">
    <div class="sec-head"><span class="sec-no">03</span><h2>客群分群與營收集中度</h2></div>
    <p class="lede">SQL <span class="mono">NTILE</span> 評分箱分六群，K-means(k=3) 獨立驗證同一結構（Champions 99% 落同群）。</p>
    <div class="split">
      <div>{sbars}</div>
      <div><div class="subcap">▸ 營收集中度（少數高價值客戶撐起多數營收）</div>{conc_html}</div>
    </div>
    <table class="tbl"><thead><tr><th>客群</th><th>人數</th><th>平均金額 BRL</th><th>平均 Recency 天</th><th>建議動作</th></tr></thead>
      <tbody>{trows}</tbody></table>
  </section>

  <section id="drv" class="blk reveal">
    <div class="sec-head"><span class="sec-no">04</span><h2>複購的真正槓桿：首單品類</h2></div>
    <p class="lede">各品類 180 天複購率（基準 {base*100:.1f}%）。<span style="color:var(--accent)">紅＝高於基準</span>、<span style="color:var(--accent-2)">綠＝低於</span>。家用／可補貨遠勝一次性嗜好。</p>
    {cbars}
  </section>

  <section id="mdl" class="blk reveal">
    <div class="sec-head"><span class="sec-no">05</span><h2>高潛力客戶挖掘 — 誠實版</h2></div>
    <div class="split">
      <div>
        <p class="lede">把複購當「在 3% 稀有正樣本中撈高潛力少數」的排序問題。下圖：依模型分數由高到低鎖定客戶，可捕獲多少複購者。</p>
        {gsvg}
      </div>
      <div>
        <div class="model-grid" style="grid-template-columns:1fr">
          <div class="mcard"><div class="mv">0.59</div><div class="ml">ROC-AUC（5 種子穩健 ±.008）</div></div>
          <div class="mcard"><div class="mv">1.8×</div><div class="ml">Top-decile lift（前 10% 捕獲 ~18% 複購者）</div></div>
          <div class="mcard"><div class="mv">2.06×</div><div class="ml">PR-AUC vs baseline（XGBoost 對照）</div></div>
        </div>
      </div>
    </div>
    <div class="aside">
      <span class="tag">誠實評估 · The Honest Read</span>
      <p>模型絕對值偏弱，這<b>不是執行失誤、而是資料的天花板</b>：僅憑首單交易資料，複購本質接近不可預測。LogReg 與 XGBoost 同落此線 → 模型非瓶頸，資料才是。</p>
      <p>我<b>沒有</b>用聚合 RFM 衝一個漂亮但洩漏的分數。經一次獨立的程式碼審查確認 pipeline <b>substantially leak-safe</b>，指標未受污染；標籤的 delivered-censoring 為保守偏誤，已揭露不掩蓋。</p>
    </div>
  </section>

  <footer class="reveal">
    <div class="eyebrow">Reproducibility & Method</div>
    <p>Python 3.13 · 隨機種子全固定 · 由 raw CSV 確定性重建（<span class="mono">00 → 08</span>）。客戶主鍵採 <span class="mono">customer_unique_id</span> 跨單追蹤；有效訂單僅取 <span class="mono">delivered</span>。完整決策鏈見 <span class="mono">logs/decisions.log</span>。</p>
    <div class="files">細部互動圖：outputs/eda_report · rfm_segments · propensity · xgb_shap (.html)<br>文件：docs/data-contract · eda-summary · experiment-tracking · evaluation-report</div>
  </footer>
</div>
"""

    html = ("<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Olist 客戶分析 — 成果報告</title>"
            "<link rel='preconnect' href='https://fonts.googleapis.com'>"
            "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
            "<link href='https://fonts.googleapis.com/css2?"
            "family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,900;1,9..144,500&"
            "family=IBM+Plex+Mono:wght@400;500;600&family=Noto+Sans+TC:wght@400;500&"
            "family=Noto+Serif+TC:wght@400;600;900&display=swap' rel='stylesheet'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")

    out = config.ROOT / "index.html"  # 唯一 HTML 交付（GitHub Pages 根連結）
    out.write_text(html, encoding="utf-8")
    print(f"客戶 {n:,}｜複購率 {repeat*100:.1f}%｜高價值 {champ_pct*100:.0f}% 客戶佔 {champ_share*100:.0f}% 營收"
          f"｜cohort 最高留存(月1-6) {maxret:.2f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
