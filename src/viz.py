"""共用視覺化工具：把多張 plotly 圖組成單一、可分享的 HTML 檔。

模塊五的成果報告也會重用這個 helper。
"""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

_CSS = (
    "body{font-family:'Noto Sans TC','Microsoft JhengHei',system-ui,sans-serif;"
    "max-width:1040px;margin:24px auto;padding:0 16px;color:#1a1a1a;line-height:1.6}"
    "h1{border-bottom:3px solid #2c3e50;padding-bottom:8px}"
    "h2{margin-top:36px;color:#2c3e50}"
    "p.note{color:#555;background:#f6f8fa;padding:10px 14px;border-left:3px solid #888}"
)


def save_report(blocks: list[tuple[str, go.Figure | str]], path: Path, title: str) -> None:
    """Write a list of (heading, figure-or-html-note) into one self-contained HTML.

    A str block is rendered as a note paragraph; a Figure is embedded as a chart.
    plotly.js is loaded once (via CDN) on the first figure only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"<h1>{title}</h1>"]
    first_fig = True
    for heading, content in blocks:
        if heading:
            parts.append(f"<h2>{heading}</h2>")
        if isinstance(content, str):
            # 以 '<' 開頭視為原始 HTML（表格等）直接嵌入，否則包成提示段落
            parts.append(content if content.lstrip().startswith("<")
                         else f"<p class='note'>{content}</p>")
        else:
            parts.append(content.to_html(
                full_html=False,
                include_plotlyjs="cdn" if first_fig else False))
            first_fig = False
    html = ("<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
            f"<title>{title}</title><style>{_CSS}</style></head>"
            f"<body>{''.join(parts)}</body></html>")
    path.write_text(html, encoding="utf-8")
