"""
05_build_dashboard.py
=====================
EXECUTIVE DASHBOARD (interactive, single self-contained HTML file)

Reads the model + scenario outputs and renders a dashboard that opens in any
browser with no server and no internet connection. Built for a CFO / Head of HR
audience: everything is framed in euros and decisions, not raw model internals.

Panels:
    1. Filters                 department, location, job level (apply to all people metrics)
    2. KPI strip               headcount, attrition, watchlist, pay gap, cost of inaction, ROI
    3. The decision            24-month cost of each scenario vs status quo
    4. Where risk sits         risk-band mix by department
    5. What drives it          model drivers, risk vs pay gap, risk vs time since last raise
    6. Retention watchlist     sortable list of the employees to act on
    7. Data quality            what the ETL step fixed or imputed

Charts are drawn as inline SVG by a small amount of plain JavaScript, so the
file has no external dependencies (no Plotly, no CDN) and stays under 100 KB.

Output:
    outputs/executive_dashboard.html

Run:  python src/05_build_dashboard.py
"""

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
DATA = ROOT / "data" / "processed"

master = pd.read_csv(DATA / "master_workforce.csv")
scored = pd.read_csv(OUT / "scored_active_employees.csv")
summary = pd.read_csv(OUT / "scenario_summary.csv")
drivers = pd.read_csv(MODELS / "feature_importance.csv")
evaluation = (OUT / "model_evaluation.txt").read_text(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Model metadata (read from the evaluation report so the dashboard never
# disagrees with the model step)
# ---------------------------------------------------------------------------
m_model = re.search(r"Selected model\s*:\s*(.+)", evaluation)
model_name = m_model.group(1).strip() if m_model else "n/a"
m_auc = re.search(re.escape(model_name) + r" \(AUC ([0-9.]+)\)", evaluation)
model_auc = float(m_auc.group(1)) if m_auc else None

# ---------------------------------------------------------------------------
# Employee-level data: one compact row per employee (active + leavers)
# ---------------------------------------------------------------------------
BANDS = ["Critical", "High", "Moderate", "Low"]
df = master.merge(scored[["employee_id", "risk_score", "risk_band", "on_watchlist"]],
                  on="employee_id", how="left")
depts = sorted(df["department"].unique())
locs = sorted(df["location"].unique())
levels = sorted(df["job_level"].unique(), key=lambda s: int(s[1]))

rows = []
for r in df.itertuples(index=False):
    has_score = pd.notna(r.risk_score)
    rows.append([
        r.employee_id,
        depts.index(r.department),
        locs.index(r.location),
        levels.index(r.job_level),
        int(bool(r.is_active)),
        int(r.attrition_flag),
        round(1 - r.comp_to_market_ratio, 3),          # pay gap: + = below market
        int(r.months_since_last_raise),
        round(float(r.risk_score), 1) if has_score else None,
        BANDS.index(r.risk_band) if has_score else None,
        int(bool(r.on_watchlist)) if has_score else 0,
        int(round(r.base_salary)),
    ])

# ---------------------------------------------------------------------------
# Company-level figures (scenario model is not split by department)
# ---------------------------------------------------------------------------
def scen(prefix):
    return summary.loc[summary["scenario"].str.startswith(prefix)].iloc[0]

A, B, C = scen("A"), scen("B"), scen("C")
scenarios = [
    {"label": "Targeted raises now", "code": "B", "delta": float(B.vs_status_quo), "total": float(B.total_cost)},
    {"label": "Status quo", "code": "A", "delta": 0.0, "total": float(A.total_cost)},
    {"label": "Do nothing, backfill", "code": "C", "delta": float(C.vs_status_quo), "total": float(C.total_cost)},
]
company = {
    "cost_of_inaction": float(C.total_cost - B.total_cost),
    "net_saving": float(A.total_cost - B.total_cost),
    "retention_budget": float(B.retention_investment),
}

data_quality = {
    "benchmark_imputed": int(master["benchmark_imputed"].sum()),
    "salary_imputed": int(master["salary_imputed"].sum()),
    "performance_imputed": int(master["performance_imputed"].sum()),
    "salary_corrected": int(master["salary_corrected"].sum()),
    "records": int(len(master)),
}

payload = {
    "rows": rows,
    "depts": depts, "locs": locs, "levels": levels, "bands": BANDS,
    "drivers": [{"name": d.driver, "pct": float(d.importance_pct)}
                for d in drivers.sort_values("importance_pct", ascending=False).itertuples()],
    "scenarios": scenarios,
    "company": company,
    "dq": data_quality,
    "model": {"name": model_name, "auc": model_auc},
    "generated": date.today().isoformat(),
}

# ---------------------------------------------------------------------------
# HTML template (CSS + JS inline)
# ---------------------------------------------------------------------------
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Workforce Attrition Risk Dashboard</title>
<style>
:root{
  --bg:#f4f5f7; --card:#ffffff; --line:#e4e3df; --grid:#ecebe7;
  --ink:#0b0b0b; --ink2:#52514e; --ink3:#8a8984; --navy:#1f3864;
  --crit:#c0392b; --high:#eda100; --mod:#2e5c8a; --low:#1baf7a;
  --save:#2a78d6; --cost:#eb6834;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI",-apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif}
header{background:var(--navy);color:#fff;padding:22px 32px}
header h1{margin:0;font-size:22px;font-weight:600;letter-spacing:-.01em}
header p{margin:4px 0 0;color:#c9d4e6;font-size:13px}
main{max-width:1320px;margin:0 auto;padding:20px 32px 40px}
.filters{display:flex;gap:12px;flex-wrap:wrap;align-items:end;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-bottom:16px}
.filters label{display:flex;flex-direction:column;font-size:11px;color:var(--ink2);text-transform:uppercase;letter-spacing:.04em;gap:4px}
.filters select{font:inherit;font-size:13px;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;min-width:170px;color:var(--ink)}
.filters button{font:inherit;font-size:13px;padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer;color:var(--ink2)}
.filters .note{margin-left:auto;font-size:12px;color:var(--ink3)}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .l{font-size:11px;color:var(--ink2);text-transform:uppercase;letter-spacing:.04em}
.kpi .v{font-size:26px;font-weight:650;color:var(--navy);margin-top:4px;letter-spacing:-.01em}
.kpi .s{font-size:12px;color:var(--ink3);margin-top:2px}
.kpi.co{border-top:3px solid var(--cost)}
.section{font-size:13px;font-weight:600;color:var(--ink2);text-transform:uppercase;letter-spacing:.05em;margin:22px 2px 10px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid3{display:grid;grid-template-columns:1fr 1.25fr 1fr;gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.card h3{margin:0;font-size:15px;font-weight:600}
.card .sub{margin:2px 0 10px;font-size:12.5px;color:var(--ink2)}
svg text{font-family:inherit}
.legend{display:flex;gap:14px;font-size:12px;color:var(--ink2);margin-top:6px;flex-wrap:wrap}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;font-weight:600;color:var(--ink2);font-size:12px;border-bottom:1px solid var(--line);padding:8px 10px;cursor:pointer;white-space:nowrap;user-select:none}
th.num,td.num{text-align:right}
th:hover{color:var(--ink)}
td{padding:7px 10px;border-bottom:1px solid var(--grid)}
tr:hover td{background:#f7f8fa}
.bar{display:inline-block;height:8px;border-radius:4px;background:var(--crit);vertical-align:middle;margin-right:8px}
.tablewrap{max-height:420px;overflow:auto}
.dq{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.dq div{border-left:3px solid var(--mod);padding:4px 12px}
.dq b{display:block;font-size:20px;color:var(--navy)}
.dq span{font-size:12.5px;color:var(--ink2)}
footer{max-width:1320px;margin:0 auto;padding:0 32px 30px;font-size:12px;color:var(--ink3)}
#tip{position:fixed;pointer-events:none;background:#0b0b0b;color:#fff;font-size:12px;padding:7px 10px;border-radius:6px;opacity:0;transition:opacity .08s;max-width:260px;z-index:10;line-height:1.4}
.empty{color:var(--ink3);font-size:13px;padding:20px 0;text-align:center}
@media (max-width:1100px){.kpis{grid-template-columns:repeat(3,1fr)}.grid3{grid-template-columns:1fr}.dq{grid-template-columns:repeat(2,1fr)}}
@media (max-width:760px){.kpis{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}main,header{padding-left:16px;padding-right:16px}}
</style>
</head>
<body>
<header>
  <h1>Workforce Cost &amp; Attrition Risk</h1>
  <p id="headsub"></p>
</header>
<main>
  <div class="filters">
    <label>Department<select id="f-dept"></select></label>
    <label>Location<select id="f-loc"></select></label>
    <label>Job level<select id="f-lvl"></select></label>
    <button id="f-reset" type="button">Reset filters</button>
    <span class="note">Filters apply to all people metrics. Cost figures are company-wide.</span>
  </div>

  <div class="kpis" id="kpis"></div>

  <div class="section">The decision</div>
  <div class="grid2">
    <div class="card">
      <h3>24-month cost vs. status quo</h3>
      <p class="sub">Company-wide. Below zero = saving, above zero = extra cost.</p>
      <div id="c-scen"></div>
    </div>
    <div class="card">
      <h3>Where risk is concentrated</h3>
      <p class="sub">Share of active employees in each risk band, sorted by High + Critical share.</p>
      <div id="c-dept"></div>
      <div class="legend" id="lg-band"></div>
    </div>
  </div>

  <div class="section">What drives it</div>
  <div class="grid3">
    <div class="card">
      <h3>Attrition drivers</h3>
      <p class="sub">Share of model importance (<span id="mname"></span>).</p>
      <div id="c-drv"></div>
    </div>
    <div class="card">
      <h3>Risk vs. pay gap to market</h3>
      <p class="sub">Each dot is an active employee. Right of the line = paid below market.</p>
      <div id="c-sc"></div>
      <div class="legend" id="lg-band2"></div>
    </div>
    <div class="card">
      <h3>Risk vs. time since last raise</h3>
      <p class="sub">Average risk score of active employees.</p>
      <div id="c-raise"></div>
    </div>
  </div>

  <div class="section">Who to act on</div>
  <div class="card">
    <h3>Retention watchlist <span id="wl-count" style="color:var(--ink3);font-weight:400"></span></h3>
    <p class="sub">Top 12% of active employees by risk. Click a column header to sort.</p>
    <div class="tablewrap"><table id="wl"></table></div>
  </div>

  <div class="section">Data quality</div>
  <div class="card">
    <div class="dq" id="dq"></div>
  </div>
</main>
<footer id="foot"></footer>
<div id="tip"></div>

<script>
const D = __DATA__;
const COL = {Critical:"#c0392b", High:"#eda100", Moderate:"#2e5c8a", Low:"#1baf7a"};
const ID=0,DEPT=1,LOC=2,LVL=3,ACT=4,LEFT=5,GAP=6,MSR=7,RISK=8,BAND=9,WATCH=10,SAL=11;
const state = {dept:-1, loc:-1, lvl:-1, sortKey:"risk", sortDir:-1};

const $ = s => document.querySelector(s);
const fmtEur = v => (Math.abs(v)>=1e6 ? "€"+(v/1e6).toFixed(v%1e6===0?0:(Math.abs(v)<1e6*10?2:1))+"M" : "€"+Math.round(v).toLocaleString("en-GB"));
const eurM = (v,d=1) => (v<0?"−":"")+"€"+(Math.abs(v)/1e6).toFixed(d)+"M";
const pct = (v,d=1) => (100*v).toFixed(d)+"%";
const esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;");

// ---------- tooltip ----------
const tip = $("#tip");
document.addEventListener("mousemove", e => {
  const t = e.target.closest("[data-tip]");
  if (!t) { tip.style.opacity = 0; return; }
  tip.innerHTML = t.getAttribute("data-tip");
  const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
  const y = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px"; tip.style.opacity = 1;
});

// ---------- filters ----------
function fillSelect(sel, items, all){
  sel.innerHTML = `<option value="-1">${all}</option>` + items.map((n,i)=>`<option value="${i}">${esc(n)}</option>`).join("");
}
fillSelect($("#f-dept"), D.depts, "All departments");
fillSelect($("#f-loc"), D.locs, "All locations");
fillSelect($("#f-lvl"), D.levels, "All levels");
["dept","loc","lvl"].forEach(k => $("#f-"+k).addEventListener("change", e => { state[k] = +e.target.value; render(); }));
$("#f-reset").addEventListener("click", () => { state.dept=state.loc=state.lvl=-1; ["dept","loc","lvl"].forEach(k=>$("#f-"+k).value="-1"); render(); });

function inFilter(r, ignoreDept){
  return (ignoreDept || state.dept<0 || r[DEPT]===state.dept) &&
         (state.loc<0 || r[LOC]===state.loc) && (state.lvl<0 || r[LVL]===state.lvl);
}

// ---------- SVG helpers ----------
const svg = (w,h,inner) => `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" style="display:block">${inner}</svg>`;
const rr = (x,y,w,h,r,fill,extra="") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0,w).toFixed(1)}" height="${h.toFixed(1)}" rx="${r}" fill="${fill}" ${extra}/>`;
const tx = (x,y,s,attrs="") => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" ${attrs}>${s}</text>`;

// ---------- KPIs ----------
function renderKpis(all, active){
  const leavers = all.filter(r=>r[LEFT]===1).length;
  const watch = active.filter(r=>r[WATCH]===1);
  const gap = watch.length ? watch.reduce((a,r)=>a+Math.max(0,r[GAP]),0)/watch.length : 0;
  const avgRisk = active.length ? active.reduce((a,r)=>a+r[RISK],0)/active.length : 0;
  const c = D.company;
  const k = [
    ["Active headcount", active.length.toLocaleString("en-GB"), `avg risk score ${avgRisk.toFixed(1)} / 100`, ""],
    ["Historical attrition", all.length? pct(leavers/all.length):"–", `${leavers} of ${all.length} employees left`, ""],
    ["Retention watchlist", watch.length, active.length? `${pct(watch.length/active.length)} of active staff`:"", ""],
    ["Watchlist pay gap", watch.length? pct(gap):"–", "average below market pay", ""],
    ["Cost of inaction", eurM(c.cost_of_inaction), "24 months, vs acting now", "co"],
    ["Return on retention", (c.cost_of_inaction/c.retention_budget).toFixed(1)+"×", `on a ${eurM(c.retention_budget,2)} raise budget`, "co"],
  ];
  $("#kpis").innerHTML = k.map(([l,v,s,cls])=>`<div class="kpi ${cls}"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
}

// ---------- scenario chart (diverging bars) ----------
function renderScen(){
  const W=560,H=210, L=170, R=90, top=16, rowH=56;
  const vals=D.scenarios.map(s=>s.delta), mx=Math.max(...vals.map(Math.abs));
  const hi=Math.max(0,...vals), lo=Math.min(0,...vals,-0.3*hi);   // leave room left of zero for the saving label
  const x = v => L + (v-lo)/(hi-lo)*(W-L-R);
  let g="";
  D.scenarios.forEach((s,i)=>{
    const y=top+i*rowH;
    g+=tx(0,y+22,esc(s.label),'font-size="13" fill="#0b0b0b"');
    g+=tx(0,y+39,"total "+eurM(s.total,1),'font-size="11.5" fill="#8a8984"');
    const x0=x(0), x1=x(s.delta), col=s.delta<0?"#2a78d6":(s.delta>0?"#eb6834":"#8a8984");
    const t=`<b>${esc(s.label)}</b><br>vs status quo: ${s.delta===0?"baseline":(s.delta>0?"+":"")+eurM(s.delta,2)}<br>24-month total: ${eurM(s.total,1)}`;
    if(s.delta!==0){ g+=rr(Math.min(x0,x1),y+12,Math.abs(x1-x0),24,4,col,`data-tip="${esc(t)}"`); }
    else { g+=rr(x0-2,y+12,4,24,1,col,`data-tip="${esc(t)}"`); }
    const lab = s.delta===0?"baseline":(s.delta>0?"+":"")+eurM(s.delta,2);
    g+= s.delta<0 ? tx(Math.min(x0,x1)-6,y+29,lab,'font-size="12.5" font-weight="600" text-anchor="end" fill="#0b0b0b"')
                  : tx(Math.max(x0,x1)+6,y+29,lab,'font-size="12.5" font-weight="600" fill="#0b0b0b"');
  });
  g+=`<line x1="${x(0)}" x2="${x(0)}" y1="${top}" y2="${top+3*rowH}" stroke="#52514e"/>`;
  const c=D.company;
  $("#c-scen").innerHTML = svg(W,H,g) +
    `<p class="sub" style="margin:4px 0 0">Spending <b>${eurM(c.retention_budget,2)}</b> on targeted raises for the watchlist saves <b>${eurM(c.net_saving,2)}</b> against the status quo and avoids <b>${eurM(c.cost_of_inaction,1)}</b> compared with backfilling leavers.</p>`;
}

// ---------- department risk mix (100% stacked) ----------
function renderDept(){
  const act = D.rows.filter(r=>r[ACT]===1 && inFilter(r,true));
  const by = D.depts.map((n,i)=>{
    const rs=act.filter(r=>r[DEPT]===i), cnt=[0,0,0,0];
    rs.forEach(r=>cnt[r[BAND]]++);
    return {n,i,tot:rs.length,cnt,hs:rs.length?(cnt[0]+cnt[1])/rs.length:0};
  }).filter(d=>d.tot>0).sort((a,b)=>b.hs-a.hs);
  if(!by.length){ $("#c-dept").innerHTML='<div class="empty">No employees match these filters.</div>'; return; }
  const W=560, L=130, R=46, rowH=27, H=by.length*rowH+24;
  let g="";
  [0,.25,.5,.75,1].forEach(t=>{ const xx=L+t*(W-L-R); g+=`<line x1="${xx}" x2="${xx}" y1="0" y2="${by.length*rowH}" stroke="#ecebe7"/>`+tx(xx,by.length*rowH+16,(t*100)+"%",'font-size="11" fill="#8a8984" text-anchor="middle"'); });
  by.forEach((d,k)=>{
    const y=k*rowH+4, dim = state.dept>=0 && d.i!==state.dept ? 'opacity="0.3"' : "";
    g+=tx(L-8,y+14,esc(d.n),`font-size="12.5" fill="#52514e" text-anchor="end" ${dim}`);
    let xx=L;
    d.cnt.forEach((c,b)=>{
      const w=c/d.tot*(W-L-R); if(!w) return;
      const t=`<b>${esc(d.n)}</b><br>${D.bands[b]}: ${c} of ${d.tot} (${pct(c/d.tot)})`;
      g+=rr(xx,y,w-(w>2?2:0),19,2,COL[D.bands[b]],`${dim} data-tip="${esc(t)}"`); xx+=w;
    });
    g+=tx(W-R+6,y+14,pct(d.hs,0),`font-size="12" font-weight="600" fill="#0b0b0b" ${dim}`);
  });
  $("#c-dept").innerHTML = svg(W,H,g);
}

// ---------- drivers ----------
function renderDrivers(){
  const ds=D.drivers, W=380, L=150, R=44, rowH=26, H=ds.length*rowH+6, mx=Math.max(...ds.map(d=>d.pct));
  let g="";
  ds.forEach((d,k)=>{
    const y=k*rowH+3, w=d.pct/mx*(W-L-R);
    const name=d.name.replace(" (comp ratio)","");
    g+=tx(L-8,y+14,esc(name),'font-size="12" fill="#52514e" text-anchor="end"');
    g+=rr(L,y+3,w,15,3,"#2a78d6",`data-tip="<b>${esc(d.name)}</b><br>${d.pct.toFixed(1)}% of model importance"`);
    g+=tx(L+w+6,y+15,d.pct.toFixed(1)+"%",'font-size="12" font-weight="600" fill="#0b0b0b"');
  });
  $("#c-drv").innerHTML = svg(W,H,g);
}

// ---------- scatter ----------
function renderScatter(act){
  const W=470,H=300,L=40,B=34,T=8,R=10;
  const xs=D.rows.filter(r=>r[ACT]===1).map(r=>r[GAP]);
  const xmin=Math.floor(Math.min(...xs)*10)/10, xmax=Math.ceil(Math.max(...xs)*10)/10;
  const X=v=>L+(v-xmin)/(xmax-xmin)*(W-L-R), Y=v=>T+(1-v/100)*(H-T-B);
  let g="";
  [0,20,40,60,80,100].forEach(v=>{ g+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#ecebe7"/>`+tx(L-6,Y(v)+4,v,'font-size="11" fill="#8a8984" text-anchor="end"'); });
  for(let v=Math.ceil(xmin*5)/5; v<=xmax+1e-9; v+=0.2){ g+=tx(X(v),H-B+16,Math.round(v*100)+"%",'font-size="11" fill="#8a8984" text-anchor="middle"'); }
  g+=`<line x1="${X(0)}" x2="${X(0)}" y1="${T}" y2="${H-B}" stroke="#52514e" stroke-dasharray="4 4"/>`;
  g+=tx(X(0)+5,T+12,"at market pay",'font-size="11" fill="#52514e"');
  g+=tx((L+W-R)/2,H-2,"pay gap to market (+ = below market)",'font-size="11.5" fill="#52514e" text-anchor="middle"');
  g+=tx(12,(T+H-B)/2,"risk score",`font-size="11.5" fill="#52514e" text-anchor="middle" transform="rotate(-90 12 ${(T+H-B)/2})"`);
  const order=[3,2,1,0];   // draw Low first so Critical sits on top
  order.forEach(b=>act.filter(r=>r[BAND]===b).forEach(r=>{
    const t=`<b>${r[ID]}</b> · ${esc(D.depts[r[DEPT]])}<br>${esc(D.levels[r[LVL]])}, ${esc(D.locs[r[LOC]])}<br>Risk ${r[RISK].toFixed(0)} (${D.bands[b]}) · pay gap ${pct(r[GAP],0)}`;
    g+=`<circle cx="${X(r[GAP]).toFixed(1)}" cy="${Y(r[RISK]).toFixed(1)}" r="4" fill="${COL[D.bands[b]]}" fill-opacity="0.8" stroke="#fff" stroke-width="1" data-tip="${esc(t)}"/>`;
  }));
  $("#c-sc").innerHTML = act.length ? svg(W,H,g) : '<div class="empty">No employees match these filters.</div>';
}

// ---------- risk by months since last raise ----------
function renderRaise(act){
  const B=[["0–6",0,6],["7–12",7,12],["13–18",13,18],["19+",19,1e9]];
  const v=B.map(([n,a,b])=>{const rs=act.filter(r=>r[MSR]>=a&&r[MSR]<=b); return {n,cnt:rs.length,avg:rs.length?rs.reduce((s,r)=>s+r[RISK],0)/rs.length:null};});
  const W=360,H=300,L=34,Bt=40,T=24,R=8;
  const mx=Math.max(50, Math.ceil(Math.max(...v.map(d=>d.avg||0))/10)*10);   // axis grows when a filter pushes risk above 50
  const Y=x=>T+(1-x/mx)*(H-T-Bt), slot=(W-L-R)/B.length, bw=Math.min(52,slot*.6);
  let g="";
  [...Array(mx/10+1).keys()].map(i=>i*10).forEach(t=>{ g+=`<line x1="${L}" x2="${W-R}" y1="${Y(t)}" y2="${Y(t)}" stroke="#ecebe7"/>`+tx(L-6,Y(t)+4,t,'font-size="11" fill="#8a8984" text-anchor="end"'); });
  v.forEach((d,i)=>{
    const cx=L+slot*i+slot/2;
    if(d.avg!==null){
      const y=Y(Math.min(d.avg,mx)), h=(H-Bt)-y;
      g+=`<path d="M${cx-bw/2},${H-Bt} V${y+4} Q${cx-bw/2},${y} ${cx-bw/2+4},${y} H${cx+bw/2-4} Q${cx+bw/2},${y} ${cx+bw/2},${y+4} V${H-Bt} Z" fill="#2a78d6" data-tip="<b>${d.n} months since last raise</b><br>Average risk ${d.avg.toFixed(1)}<br>${d.cnt} employees"/>`;
      g+=tx(cx,y-6,d.avg.toFixed(1),'font-size="12" font-weight="600" fill="#0b0b0b" text-anchor="middle"');
    }
    g+=tx(cx,H-Bt+16,d.n,'font-size="12" fill="#52514e" text-anchor="middle"');
  });
  g+=tx((L+W-R)/2,H-6,"months since last raise",'font-size="11.5" fill="#52514e" text-anchor="middle"');
  $("#c-raise").innerHTML = svg(W,H,g);
}

// ---------- watchlist ----------
const COLS=[
  {k:"id",h:"Employee",f:r=>r[ID],v:r=>r[ID]},
  {k:"dept",h:"Department",f:r=>esc(D.depts[r[DEPT]]),v:r=>D.depts[r[DEPT]]},
  {k:"lvl",h:"Level",f:r=>esc(D.levels[r[LVL]]),v:r=>r[LVL]},
  {k:"loc",h:"Location",f:r=>esc(D.locs[r[LOC]]),v:r=>D.locs[r[LOC]]},
  {k:"sal",h:"Base salary",num:1,f:r=>"€"+r[SAL].toLocaleString("en-GB"),v:r=>r[SAL]},
  {k:"gap",h:"Pay gap",num:1,f:r=>pct(r[GAP],0),v:r=>r[GAP]},
  {k:"msr",h:"Months since raise",num:1,f:r=>r[MSR],v:r=>r[MSR]},
  {k:"risk",h:"Risk",f:r=>`<span class="bar" style="width:${(r[RISK]*0.9).toFixed(0)}px;background:${COL[D.bands[r[BAND]]]}"></span><b>${r[RISK].toFixed(0)}</b>`,v:r=>r[RISK]},
];
function renderWatch(act){
  const rs=act.filter(r=>r[WATCH]===1);
  const c=COLS.find(c=>c.k===state.sortKey);
  rs.sort((a,b)=>{const x=c.v(a),y=c.v(b); return (x<y?-1:x>y?1:0)*state.sortDir;});
  $("#wl-count").textContent = `· ${rs.length} employee${rs.length===1?"":"s"}`;
  const head="<thead><tr>"+COLS.map(c=>`<th class="${c.num?"num":""}" data-k="${c.k}">${c.h}${state.sortKey===c.k?(state.sortDir<0?" ▾":" ▴"):""}</th>`).join("")+"</tr></thead>";
  const body=rs.length? rs.map(r=>"<tr>"+COLS.map(c=>`<td class="${c.num?"num":""}">${c.f(r)}</td>`).join("")+"</tr>").join("")
                      : `<tr><td colspan="${COLS.length}" class="empty">No watchlist employees match these filters.</td></tr>`;
  $("#wl").innerHTML=head+"<tbody>"+body+"</tbody>";
  $("#wl").querySelectorAll("th").forEach(th=>th.addEventListener("click",()=>{
    const k=th.dataset.k; if(state.sortKey===k) state.sortDir*=-1; else {state.sortKey=k; state.sortDir=(k==="risk"||k==="gap"||k==="sal"||k==="msr")?-1:1;} renderWatch(act);
  }));
}

// ---------- static parts ----------
function renderStatic(){
  const m=D.model, act=D.rows.filter(r=>r[ACT]===1).length;
  $("#headsub").textContent = `People analytics · ${act} active employees · 24-month financial horizon · synthetic data`;
  $("#mname").textContent = m.name + (m.auc? `, ROC-AUC ${m.auc.toFixed(3)}`:"");
  const lg = D.bands.map(b=>`<span><i style="background:${COL[b]}"></i>${b}</span>`).join("");
  $("#lg-band").innerHTML = lg; $("#lg-band2").innerHTML = lg;
  const q=D.dq;
  $("#dq").innerHTML = [
    [q.records, "employee records integrated from HRIS, payroll and benchmark files"],
    [q.benchmark_imputed, `missing market benchmarks (${pct(q.benchmark_imputed/q.records,0)}) filled from internal salary-band midpoints`],
    [q.salary_imputed + q.salary_corrected, `salaries imputed (${q.salary_imputed}) or corrected (${q.salary_corrected})`],
    [q.performance_imputed, "missing performance ratings imputed and flagged"],
  ].map(([n,t])=>`<div><b>${n}</b><span>${t}</span></div>`).join("");
  $("#foot").innerHTML = `Model: ${esc(m.name)}${m.auc?` (held-out ROC-AUC ${m.auc.toFixed(3)})`:""} · risk scores are predicted probabilities ×100 · every imputed value keeps an audit flag in the data · synthetic data for portfolio demonstration · generated ${D.generated} by src/05_build_dashboard.py`;
  renderScen(); renderDrivers();
}

function render(){
  const all=D.rows.filter(r=>inFilter(r));
  const act=all.filter(r=>r[ACT]===1);
  renderKpis(all,act); renderDept(); renderScatter(act); renderRaise(act); renderWatch(act);
}
renderStatic(); render();
</script>
</body>
</html>
"""

html = TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
(OUT / "executive_dashboard.html").write_text(html, encoding="utf-8")
print(f"Dashboard written to {OUT / 'executive_dashboard.html'} ({len(html)/1024:.0f} KB)")
