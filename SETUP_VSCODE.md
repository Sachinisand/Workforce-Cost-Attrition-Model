# How to Run This in VS Code (step-by-step)

This guide gets the project running on your machine in VS Code and walks you
through demoing it live. No prior setup assumed beyond having Python installed.

---

## Part 1 — One-time setup (about 5 minutes)

### 1. Install the prerequisites (if you don't already have them)
- **Python 3.10 or newer** — check by running `python --version` in a terminal.
  If it's missing, install from https://www.python.org/downloads (on Windows,
  tick **"Add Python to PATH"** during install).
- **VS Code** — https://code.visualstudio.com
- In VS Code, install the **Python extension** by Microsoft (Extensions panel,
  the four-squares icon on the left, search "Python", Install).

### 2. Open the project
- Unzip `workforce_analytics.zip` somewhere easy to find (e.g. your Desktop).
- In VS Code: **File → Open Folder…** and select the `workforce_analytics` folder.
  (Open the *folder*, not an individual file — this makes the relative paths work.)

### 3. Open the integrated terminal
- **Terminal → New Terminal** (or `` Ctrl+` ``). A terminal opens at the bottom,
  already inside the project folder.

### 4. Create a virtual environment (keeps this project's packages tidy)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> If PowerShell blocks the activate script, run this once, then retry:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You'll know it worked when your terminal prompt shows `(.venv)` at the start.

> VS Code will usually pop up a toast asking "We noticed a new virtual
> environment — set it as the workspace default?" → click **Yes**. If not, press
> `Ctrl+Shift+P`, type "Python: Select Interpreter", and pick the one with
> `.venv` in the path.

### 5. Install the libraries
```bash
pip install -r requirements.txt
```
This pulls in pandas, numpy, scikit-learn, openpyxl, and plotly. Takes a minute.

---

## Part 2 — Run it

### Option A — run everything at once (recommended for a live demo)
```bash
python run_all.py
```
This runs all five steps in order and prints progress. When it finishes, it tells
you where the dashboard is.

### Option B — run each step yourself (better for *explaining* the pipeline)
Run these one at a time, so you can talk through each stage as its output appears:
```bash
python src/01_generate_synthetic_data.py
python src/02_etl_integration.py
python src/03_attrition_model.py
python src/04_compensation_scenario_model.py
python src/05_build_dashboard.py
```

> **Tip:** in VS Code you can also open any script and click the ▷ **Run** button
> in the top-right. Just make sure step 01 has run before 02, and so on — each
> step reads the previous step's output.

### See the results
- **Dashboard:** in the VS Code Explorer (left panel), expand `outputs/`, right-click
  `executive_dashboard.html` → **Reveal in File Explorer / Finder**, then
  double-click it to open in your browser. *(Opening it from the file system, not
  VS Code's preview, gives you the full interactive charts.)*
- **Excel model:** double-click `outputs/compensation_scenario_model.xlsx`.
- **Audit trail & metrics:** open `outputs/data_quality_report.txt` and
  `outputs/model_evaluation.txt` right inside VS Code.

---

## Part 3 — Demoing it well (suggested 5-minute flow)

A strong order to walk an interviewer through, all inside VS Code:

1. **Start with the problem.** Open the three files in `data/raw/` side by side.
   Point out the *different ID formats* and that they don't line up — "this is the
   integration problem the role is about."
2. **Run the ETL** (`02`) and open `outputs/data_quality_report.txt`. Talk through
   the issues it caught and fixed — this shows governance thinking, not just analysis.
3. **Run the model** (`03`) and read the top of `outputs/model_evaluation.txt`.
   Explain the ROC-AUC in the plain-English terms already written there, and that you
   compared two models and guarded against leakage.
4. **Run the scenario model** (`04`), then open the Excel file. Change one blue input
   cell on the **Assumptions** tab (e.g. the retention raise %) and show the
   **Scenario Summary** totals recompute — proves it's a real model, not a static table.
5. **Open the dashboard** and land the headline: targeted retention spend vs the much
   larger cost of doing nothing, in euros.

### Two questions you'll likely get, and where the answers live
- *"Why this model / how good is it really?"* → `outputs/model_evaluation.txt`
  (metrics, the two-model comparison, the early-warning operating point).
- *"How did you handle messy / missing data?"* → `outputs/data_quality_report.txt`
  (every issue, severity, and remediation; plus the `*_imputed` audit flags in
  `data/processed/master_workforce.csv`).

### One honest line that lands well
If asked how solid the numbers are: *"The data is synthetic, so the exact euro
figures are illustrative — but the pipeline, the governance, and the modelling are
real and would run the same way on production data. I deliberately built the signal
to be realistic, not perfect, which is why the model sits in the mid-0.80s rather
than a suspicious 0.99."*

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python: command not found` | Try `python3` instead of `python`. |
| `pip: command not found` | Use `python -m pip install -r requirements.txt`. |
| PowerShell won't activate the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry step 4. |
| A specific package version won't install | Open `requirements.txt`, delete the `==x.y.z` after that package name, save, re-run the install. |
| Dashboard charts look blank in VS Code's preview | Open the `.html` from your file explorer / browser instead — VS Code's built-in preview can't run the chart scripts. |
| "File not found" when running a script | Make sure you opened the *folder* in VS Code (Part 1, step 2) and that you ran the earlier steps first. |
| Numbers differ slightly from the README | Expected — regenerating the synthetic data reshuffles it a little. The structure and conclusions stay the same. |

---

## Optional — put it on GitHub (nice for sharing a link)

If you want to send a repo link instead of a zip:
```bash
git init
git add .
git commit -m "Workforce cost & attrition risk model — people analytics build"
```
Then create an empty repo on GitHub and follow its "push an existing repository"
instructions. The included `.gitignore` keeps the virtual environment and the
internal answer-key file out of the commit.
