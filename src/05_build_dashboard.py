"""
05_build_dashboard.py
=====================
EXECUTIVE DASHBOARD (interactive, single self-contained HTML file)

Reads the model + scenario outputs and renders a boardroom-ready dashboard that
opens in any browser with no server. Built for a CFO / Head of HR audience:
everything is framed in euros and decisions, not raw model internals.

Panels:
    1. KPI strip               headcount, watchlist size, cost-of-inaction, net saving
    2. Attrition risk by dept  stacked risk bands -> where the exposure sits
    3. Cost of inaction vs intervention  the core financial decision, in EUR
    4. Top attrition drivers   what's actually moving risk (plain-language labels)
    5. Retention watchlist     the named, actionable list with risk + comp gap

Output:
    outputs/executive_dashboard.html

Run:  python src/05_build_dashboard.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODELS = ROOT / "models"

scored = pd.read_csv(OUT / "scored_active_employees.csv")
watch = pd.read_csv(OUT / "retention_watchlist.csv")
summary = pd.read_csv(OUT / "scenario_summary.csv")
drivers = pd.read_csv(MODELS / "feature_importance.csv")

# Palette
NAVY = "#1F3864"; STEEL = "#2E5C8A"; TEAL = "#2A9D8F"
AMBER = "#E9A23B"; RED = "#C0392B"; SLATE = "#5D6D7E"
GREY = "#AEB6BF"; LIGHT = "#EAF0F6"
BAND_COLORS = {"Critical": RED, "High": AMBER, "Moderate": STEEL, "Low": TEAL}

total_A = summary.loc[summary.scenario.str.startswith("A"), "total_cost"].iloc[0]
total_B = summary.loc[summary.scenario.str.startswith("B"), "total_cost"].iloc[0]
total_C = summary.loc[summary.scenario.str.startswith("C"), "total_cost"].iloc[0]
net_saving_BA = total_A - total_B
cost_inaction = total_C - total_B

n_active = len(scored)
n_watch = len(watch)
avg_gap = (1 - watch["comp_to_market_ratio"]).clip(lower=0).mean() * 100

def eur(x, mm=False):
    if mm: return f"€{x/1e6:.1f}M"
    return f"€{x:,.0f}"

# ---------------------------------------------------------------------------
# Chart 1: Attrition risk distribution by department (stacked counts)
# ---------------------------------------------------------------------------
band_order = ["Critical", "High", "Moderate", "Low"]
pivot = (scored.groupby(["department", "risk_band"]).size()
         .unstack(fill_value=0).reindex(columns=band_order, fill_value=0))
# order departments by critical+high share
pivot["_risk"] = pivot["Critical"] + pivot["High"]
pivot = pivot.sort_values("_risk", ascending=True).drop(columns="_risk")

fig_dept = go.Figure()
for band in band_order:
    fig_dept.add_bar(
        y=pivot.index, x=pivot[band], name=band, orientation="h",
        marker_color=BAND_COLORS[band],
        hovertemplate="%{y}<br>" + band + ": %{x} employees<extra></extra>")
fig_dept.update_layout(
    barmode="stack", title="Attrition Risk by Department (active headcount)",
    height=420, plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", y=-0.15), margin=dict(l=10, r=20, t=50, b=10),
    font=dict(family="Segoe UI, Arial", size=12, color=NAVY))
fig_dept.update_xaxes(title="Employees", gridcolor=LIGHT)

# ---------------------------------------------------------------------------
# Chart 2: Cost of inaction vs intervention (scenario waterfall-style bars)
# ---------------------------------------------------------------------------
scen_labels = ["B. Proactive<br>Retention", "A. Status<br>Quo", "C. Reactive<br>(Do Nothing)"]
scen_totals = [total_B, total_A, total_C]
scen_colors = [TEAL, SLATE, RED]
fig_cost = go.Figure()
fig_cost.add_bar(
    x=scen_labels, y=scen_totals, marker_color=scen_colors,
    text=[eur(v, mm=True) for v in scen_totals], textposition="outside",
    hovertemplate="%{x}<br>Total 24m cost: %{y:,.0f} EUR<extra></extra>")
fig_cost.add_annotation(
    x="C. Reactive<br>(Do Nothing)", y=total_C,
    ax="B. Proactive<br>Retention", ay=total_B,
    xref="x", yref="y", axref="x", ayref="y",
    text=f"Cost of inaction: {eur(cost_inaction, mm=True)}",
    showarrow=True, arrowhead=3, arrowcolor=RED, font=dict(color=RED, size=13),
    bgcolor="white", bordercolor=RED)
fig_cost.update_layout(
    title="24-Month Workforce Cost: Cost of Inaction vs Intervention",
    height=420, plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=10, r=20, t=50, b=10), showlegend=False,
    font=dict(family="Segoe UI, Arial", size=12, color=NAVY),
    yaxis=dict(title="Total cost (EUR)", gridcolor=LIGHT,
               range=[min(scen_totals)*0.985, max(scen_totals)*1.02]))

# ---------------------------------------------------------------------------
# Chart 3: Top attrition drivers
# ---------------------------------------------------------------------------
dv = drivers.sort_values("importance_pct", ascending=True)
fig_drv = go.Figure(go.Bar(
    y=dv["driver"], x=dv["importance_pct"], orientation="h",
    marker_color=STEEL,
    text=[f"{v:.0f}%" for v in dv["importance_pct"]], textposition="outside",
    hovertemplate="%{y}: %{x:.1f}% of model signal<extra></extra>"))
fig_drv.update_layout(
    title="What Drives Attrition Risk (share of model signal)",
    height=420, plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=10, r=30, t=50, b=10),
    font=dict(family="Segoe UI, Arial", size=12, color=NAVY))
fig_drv.update_xaxes(title="Share of predictive signal (%)", gridcolor=LIGHT,
                     range=[0, dv["importance_pct"].max()*1.18])

# ---------------------------------------------------------------------------
# Chart 4: Risk vs pay gap scatter (the actionable insight)
# ---------------------------------------------------------------------------
scored["pay_gap_pct"] = (1 - scored["comp_to_market_ratio"]) * 100
fig_sc = go.Figure()
for band in band_order:
    sub = scored[scored["risk_band"] == band]
    fig_sc.add_trace(go.Scatter(
        x=sub["pay_gap_pct"], y=sub["risk_score"], mode="markers",
        name=band, marker=dict(color=BAND_COLORS[band], size=7, opacity=0.7,
                               line=dict(width=0.5, color="white")),
        customdata=sub[["employee_id", "department", "job_level"]],
        hovertemplate=("Risk %{y:.0f}/100<br>Pay gap %{x:.0f}%<br>"
                       "%{customdata[0]} · %{customdata[1]}<extra></extra>")))
fig_sc.add_vline(x=0, line_dash="dash", line_color=GREY)
fig_sc.update_layout(
    title="Flight Risk vs Pay Gap to Market (each dot = an employee)",
    height=420, plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", y=-0.18), margin=dict(l=10, r=20, t=50, b=10),
    font=dict(family="Segoe UI, Arial", size=12, color=NAVY))
fig_sc.update_xaxes(title="Pay gap vs market (% below market →)", gridcolor=LIGHT)
fig_sc.update_yaxes(title="Attrition risk score (0–100)", gridcolor=LIGHT)

# ---------------------------------------------------------------------------
# Watchlist table (top 15 shown; full list in CSV)
# ---------------------------------------------------------------------------
wl = watch.head(15).copy()
wl["pay_gap"] = ((1 - wl["comp_to_market_ratio"]) * 100).round(0).astype(int).astype(str) + "%"
wl["base_salary_f"] = wl["base_salary"].apply(lambda v: f"€{v:,.0f}")
wl["risk_f"] = wl["risk_score"].round(0).astype(int)
table_rows = "".join(
    f"<tr><td>{r.employee_id}</td><td>{r.department}</td><td>{r.job_level}</td>"
    f"<td style='text-align:right'>{r.base_salary_f}</td>"
    f"<td style='text-align:right'>{r.pay_gap}</td>"
    f"<td style='text-align:right'>{r.months_since_last_raise:.0f}</td>"
    f"<td style='text-align:center'><span class='pill' style='background:{RED if r.risk_f>=70 else AMBER}'>{r.risk_f}</span></td></tr>"
    for r in wl.itertuples())

# ---------------------------------------------------------------------------
# Assemble HTML
# ---------------------------------------------------------------------------
def div(fig, first=False):
    # First chart inlines the full Plotly library so the dashboard is a single
    # self-contained file that works with no internet (safe for a live interview
    # demo on any machine). Subsequent charts reuse the already-loaded library.
    return pio.to_html(fig, include_plotlyjs=("inline" if first else False),
                       full_html=False, config={"displayModeBar": False})

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Workforce Cost & Attrition Risk — Executive Dashboard</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:'Segoe UI',Arial,sans-serif; background:#F4F6F9; color:{NAVY}; }}
  header {{ background:linear-gradient(135deg,{NAVY},{STEEL}); color:white; padding:26px 40px; }}
  header h1 {{ margin:0; font-size:25px; font-weight:600; }}
  header p {{ margin:6px 0 0; opacity:.85; font-size:14px; }}
  .wrap {{ max-width:1280px; margin:0 auto; padding:24px 28px 60px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:18px; margin:24px 0 8px; }}
  .kpi {{ background:white; border-radius:12px; padding:20px 22px; box-shadow:0 2px 10px rgba(31,56,100,.08);
          border-left:5px solid {STEEL}; }}
  .kpi.alert {{ border-left-color:{RED}; }}
  .kpi.good {{ border-left-color:{TEAL}; }}
  .kpi .label {{ font-size:12.5px; color:{SLATE}; text-transform:uppercase; letter-spacing:.4px; }}
  .kpi .value {{ font-size:27px; font-weight:700; margin-top:6px; }}
  .kpi .sub {{ font-size:12px; color:{SLATE}; margin-top:3px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:8px; }}
  .card {{ background:white; border-radius:12px; padding:10px 14px 6px;
           box-shadow:0 2px 10px rgba(31,56,100,.08); }}
  .full {{ grid-column:1/3; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:{NAVY}; color:white; padding:10px 12px; text-align:left; font-weight:600; }}
  td {{ padding:9px 12px; border-bottom:1px solid #EDF1F6; }}
  tr:hover td {{ background:{LIGHT}; }}
  .pill {{ color:white; padding:3px 11px; border-radius:11px; font-weight:700; font-size:12.5px; }}
  .section-title {{ font-size:15px; font-weight:600; margin:26px 4px 4px; }}
  .note {{ font-size:12px; color:{SLATE}; margin:4px 4px 0; }}
  footer {{ text-align:center; color:{SLATE}; font-size:12px; padding:24px; }}
</style></head>
<body>
<header>
  <h1>Workforce Cost &amp; Attrition Risk — Executive Dashboard</h1>
  <p>People analytics build · {n_active} active employees · 24-month financial horizon · figures in EUR</p>
</header>
<div class="wrap">

  <div class="kpis">
    <div class="kpi">
      <div class="label">Active Headcount</div>
      <div class="value">{n_active}</div>
      <div class="sub">across {scored['department'].nunique()} departments</div>
    </div>
    <div class="kpi alert">
      <div class="label">Retention Watchlist</div>
      <div class="value">{n_watch}</div>
      <div class="sub">{n_watch/n_active:.0%} of workforce · avg {avg_gap:.0f}% below market</div>
    </div>
    <div class="kpi alert">
      <div class="label">Cost of Inaction (24m)</div>
      <div class="value">{eur(cost_inaction, mm=True)}</div>
      <div class="sub">Reactive vs Proactive path</div>
    </div>
    <div class="kpi good">
      <div class="label">Net Saving — Proactive</div>
      <div class="value">{eur(net_saving_BA, mm=True)}</div>
      <div class="sub">vs status quo, after raise spend</div>
    </div>
  </div>

  <div class="section-title">Where the risk sits, and what it costs</div>
  <div class="grid2">
    <div class="card">{div(fig_cost, first=True)}</div>
    <div class="card">{div(fig_dept)}</div>
  </div>

  <div class="section-title">What's driving it</div>
  <div class="grid2">
    <div class="card">{div(fig_drv)}</div>
    <div class="card">{div(fig_sc)}</div>
  </div>
  <p class="note">The pay-gap view is the actionable story: a cluster of high-risk
     employees sit well below market — exactly the population a targeted retention
     raise is designed to address.</p>

  <div class="section-title">Retention watchlist — top 15 (full list in retention_watchlist.csv)</div>
  <div class="card full" style="padding:0;overflow:hidden;border-radius:12px;">
    <table>
      <thead><tr>
        <th>Employee</th><th>Department</th><th>Level</th>
        <th style="text-align:right">Base Salary</th>
        <th style="text-align:right">Pay Gap</th>
        <th style="text-align:right">Mo. Since Raise</th>
        <th style="text-align:center">Risk</th>
      </tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>

  <footer>
    Model: {('Logistic Regression')} · held-out ROC-AUC ≈ 0.85 ·
    risk scores are model-estimated probabilities ×100 ·
    synthetic data, for portfolio demonstration.
  </footer>
</div>
</body></html>"""

(OUT / "executive_dashboard.html").write_text(html, encoding="utf-8")
print(f"Wrote executive_dashboard.html ({len(html)//1024} KB)")
print(f"KPIs: headcount={n_active}, watchlist={n_watch}, "
      f"cost_of_inaction={eur(cost_inaction)}, net_saving={eur(net_saving_BA)}")
