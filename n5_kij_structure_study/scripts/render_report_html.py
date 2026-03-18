#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import markdown
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from weasyprint import HTML


DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", re.S)
DISPLAY_DOLLAR_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.S)


def normalize_markdown_math(md_text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        return f"\n\n$$\n{inner}\n$$\n\n"

    return DISPLAY_RE.sub(repl, md_text)


def sanitize_latex(expr: str) -> str:
    expr = " ".join(expr.strip().split())
    expr = expr.replace(r"\|", "|")
    expr = expr.replace(r"\mathsf T", "T")
    expr = expr.replace(r"\mathsf{T}", "T")
    expr = expr.replace(r"\arg\max", r"\mathrm{argmax}")

    def text_repl(match: re.Match[str]) -> str:
        content = match.group(1)
        content = content.replace(r"\_", "")
        content = content.replace("_", "")
        content = re.sub(r"[^A-Za-z0-9]+", "", content)
        return r"\mathrm{" + content + "}"

    expr = re.sub(r"\\text\{([^}]*)\}", text_repl, expr)
    expr = re.sub(r"\\mathrm\{([^}]*)\}", text_repl, expr)
    return expr


def render_formula_svg(expr: str, asset_dir: Path, display: bool) -> str:
    asset_dir.mkdir(parents=True, exist_ok=True)
    safe_expr = sanitize_latex(expr)
    digest = hashlib.sha1((safe_expr + str(display)).encode("utf-8")).hexdigest()[:16]
    out_name = f"math_{'display' if display else 'inline'}_{digest}.svg"
    out_path = asset_dir / out_name
    if not out_path.exists():
        fontsize = 18 if display else 14
        prop = FontProperties(size=fontsize)
        math_to_image(
            f"${safe_expr}$",
            str(out_path),
            dpi=220,
            format="svg",
            prop=prop,
        )
    rel = f".formula_assets/{out_name}"
    css_class = "math-display" if display else "math-inline"
    return f'<img class="math {css_class}" src="{rel}" alt="{expr}">'


def replace_math_with_placeholders(md_text: str, asset_dir: Path) -> str:
    placeholders: list[str] = []

    def stash(expr: str, display: bool) -> str:
        idx = len(placeholders)
        placeholders.append(render_formula_svg(expr, asset_dir, display))
        return f"@@MATH{idx}@@"

    text = DISPLAY_DOLLAR_RE.sub(lambda m: stash(m.group(1), True), md_text)
    text = INLINE_RE.sub(lambda m: stash(m.group(1), False), text)

    html_body = markdown.markdown(
        text,
        extensions=[
            "extra",
            "tables",
            "fenced_code",
            "sane_lists",
            "toc",
        ],
        output_format="html5",
    )
    for idx, html in enumerate(placeholders):
        html_body = html_body.replace(f"@@MATH{idx}@@", html)
    return html_body


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    report_md = results_dir / "FIGURE_REPORT.md"
    report_html = results_dir / "FIGURE_REPORT.html"
    report_pdf = results_dir / "FIGURE_REPORT.pdf"
    asset_dir = results_dir / ".formula_assets"

    md_text = report_md.read_text(encoding="utf-8")
    normalized_md = normalize_markdown_math(md_text)
    if normalized_md != md_text:
        report_md.write_text(normalized_md, encoding="utf-8")
        md_text = normalized_md

    body = replace_math_with_placeholders(md_text, asset_dir)

    figure_paths = [
        ("Analysis Overview", "figures/analysis_overview.png"),
        ("Feature Profiles", "figures/feature_profiles.png"),
        ("Representative Kij Gallery", "figures/representative_kij_gallery.png"),
        ("Top-1 Verification", "figures/n5_top1_verification_en.png"),
    ]
    gallery_blocks = []
    for title, rel_path in figure_paths:
        abs_path = results_dir / rel_path
        if abs_path.exists():
            gallery_blocks.append(
                f"""
                <section class="figure-card">
                  <h2>{title}</h2>
                  <img class="figure-img" src="{rel_path}" alt="{title}">
                </section>
                """
            )

    gallery_html = "\n".join(gallery_blocks)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Figure Report</title>
  <style>
    :root {{
      --bg: #f6f2ea;
      --panel: #fffdf8;
      --ink: #22201c;
      --muted: #625a50;
      --line: #dfd6c9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 32px;
      background: linear-gradient(180deg, #f4efe6 0%, #fbf8f2 100%);
      color: var(--ink);
      font: 17px/1.7 Georgia, "Times New Roman", serif;
    }}
    .page {{
      max-width: 1080px;
      margin: 0 auto;
    }}
    .content {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 40px 48px;
      box-shadow: 0 10px 30px rgba(40, 30, 20, 0.08);
    }}
    h1, h2, h3, h4 {{
      color: #1f1d19;
      line-height: 1.25;
      margin-top: 1.5em;
    }}
    h1 {{
      margin-top: 0;
      font-size: 2.2rem;
      border-bottom: 2px solid var(--line);
      padding-bottom: 0.5rem;
    }}
    h2 {{
      font-size: 1.55rem;
    }}
    code {{
      background: #f2ece3;
      border-radius: 6px;
      padding: 0.12rem 0.35rem;
      font-size: 0.95em;
    }}
    pre {{
      background: #f2ece3;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow-x: auto;
      padding: 16px;
    }}
    .gallery {{
      display: grid;
      gap: 28px;
      margin-top: 32px;
    }}
    .figure-card {{
      background: #fffdfa;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px;
    }}
    .figure-card h2 {{
      margin-top: 0;
      margin-bottom: 16px;
    }}
    .figure-img {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
    }}
    .math {{
      vertical-align: middle;
    }}
    .math-inline {{
      display: inline-block;
      height: 1.3em;
    }}
    .math-display {{
      display: block;
      max-width: 100%;
      margin: 0.9rem auto;
      height: auto;
    }}
    .muted {{
      color: var(--muted);
    }}
    @page {{
      size: A4;
      margin: 18mm 16mm;
    }}
  </style>
</head>
<body>
  <div class="page">
    <article class="content">
      {body}
      <h1>Rendered Figures</h1>
      <p class="muted">The images below are embedded directly from the local results directory.</p>
      <div class="gallery">
        {gallery_html}
      </div>
    </article>
  </div>
</body>
</html>
"""

    report_html.write_text(html, encoding="utf-8")
    HTML(string=html, base_url=str(results_dir)).write_pdf(str(report_pdf))
    print(report_html)
    print(report_pdf)


if __name__ == "__main__":
    main()
