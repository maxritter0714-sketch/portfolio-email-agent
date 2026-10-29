# Portfolio Email Agent

Sends you a polished HTML portfolio report every **other Saturday**.  
The report includes live prices, P&L, benchmark comparison, curated news, and a four-section AI analysis from Claude.

> **Heads up:** This project was built for personal use and may require some adaptation before it works out of the box for you. In particular:
> - **ISIN → ticker overrides** live in `data/isin_map.json` (gitignored, not included). Copy `data/isin_map.json.example` to get started — most ISINs are resolved automatically via OpenFIGI
> - The **sector and region classification** in `market.py` is resolved dynamically from yfinance — no hardcoded maps to maintain
> - The **Finanzfluss import script** expects column names from a Finanzfluss export — other brokers will need column name adjustments
> - The agent is built around **EUR as base currency**

---

## Prerequisites

- Python 3.11 or newer
- A Gmail account with 2-Step Verification enabled

---

## 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2 — Get your API keys

### Anthropic API key
1. Go to [console.anthropic.com](https://console.anthropic.com) → **API Keys**
2. Click **Create Key**, copy it

### NewsAPI key
1. Go to [newsapi.org/register](https://newsapi.org/register)
2. Sign up for a free account, copy your API key from the dashboard

### Gmail App Password
Gmail requires an **App Password** instead of your regular password when using SMTP with 2-Step Verification.

1. Go to your [Google Account](https://myaccount.google.com) → **Security**
2. Under "How you sign in to Google", open **2-Step Verification** (enable it if not already on)
3. Scroll to the bottom → **App passwords**
4. Select app: **Mail**, device: **Other** → name it "Portfolio Agent"
5. Copy the 16-character password (no spaces needed)

---

## 3 — Configure your secrets

Copy the example file and fill it in:

```bash
cp .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
NEWS_API_KEY=abc123...
GMAIL_ADDRESS=you@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
RECIPIENT_EMAIL=you@gmail.com
```

> **Security:** Never commit `.env` to version control. Add it to `.gitignore`.

---

## 4 — Set up your portfolio

> **Finanzfluss user?** Skip the manual edit below and use the [import script](#7--update-your-holdings-finanzfluss-users) (step 7) instead — it generates `data/portfolio.csv` for you.

All personal files live in `data/` — gitignored except the `*.example` templates — and logs and previews go to `output/`.

Copy `data/portfolio.csv.example` to `data/portfolio.csv` and fill in your real holdings:

```
ticker,name,shares,avg_buy_price
AAPL,Apple,10,150.00
NVDA,Nvidia,5,480.00
```

**Column notes:**
- `ticker` — Yahoo Finance symbol (e.g. `AAPL`, `MSFT`, `BTC-USD`)
- `name` — Human-readable name used for news searches (e.g. `Apple`, `Nvidia`)
- `shares` — Quantity you hold
- `avg_buy_price` — Your average purchase price **in EUR**

> If your brokerage shows prices in USD, convert your buy prices to EUR at the rate you paid.

---

## 5 — Test it

Build the report locally without sending anything, using `--preview` (writes `output/preview.html` instead of emailing — no Gmail setup required for this step):

```bash
python agent.py --preview
```

Once that looks right, run with `--force` to bypass the fortnight schedule check and send a real email:

```bash
python agent.py --force
```

You should receive an email within about 30 seconds. Check `output/agent.log` if something goes wrong.

---

## 6 — Schedule it

### GitHub Actions (recommended)

The cleanest setup — runs in the cloud, no local machine required, secrets stored encrypted.

1. Create a private GitHub repo (e.g. `portfolio-agent-data`) and upload your holdings there: either `finanzfluss_export.csv` (converted on every run) or a hand-kept `portfolio.csv` (used as is). `history.csv` and `conviction.json` are created by the first run — only upload them if you already have them from local runs. If you use Finanzfluss import overrides, also upload your `isin_map.json` (see [step 7](#7--update-your-holdings-finanzfluss-users)) — otherwise it won't be picked up by cloud runs
2. Create a fine-grained Personal Access Token scoped to that private repo with **Contents: Read and write**
3. Add secrets to your public repo under **Settings → Secrets and variables → Actions**:
   - `DATA_REPO_TOKEN` — the PAT from step 2
   - `ANTHROPIC_API_KEY`, `NEWS_API_KEY`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`, `RECIPIENT_EMAIL`
4. If your private data repo is not `maxritter0714-sketch/portfolio-agent-data`, add an Actions **variable** `PORTFOLIO_DATA_REPO` set to `owner/name`.
5. The included `.github/workflows/report.yml` runs every Saturday at 6am UTC automatically

The workflow collects a weekly data point on every Saturday and sends the full email report every other Saturday. `history.csv` and `conviction.json` are pushed back to the private repo after each run so they persist between executions.

To trigger a test run: **Actions → Portfolio Report → Run workflow → tick Force full report**.

### Mac / Linux (cron)

Open your crontab:

```bash
crontab -e
```

Add this line to run at 08:00 on every Saturday (the script self-checks for a fortnight send Saturday):

```cron
0 8 * * 6 cd /path/to/portfolioAgent && /usr/bin/python3 agent.py >> output/agent.log 2>&1
```

Replace `/path/to/portfolioAgent` with the full path to this directory and `/usr/bin/python3` with the output of `which python3`.

For weekly data collection without sending an email:

```cron
0 8 * * 6 cd /path/to/portfolioAgent && /usr/bin/python3 agent.py --data-only >> output/agent.log 2>&1
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `data/portfolio.csv does not exist` | Make sure you're running the script from the `portfolioAgent` directory |
| `No price history returned` | Check the ticker symbol on [finance.yahoo.com](https://finance.yahoo.com) |
| Gmail authentication error | Re-generate the App Password; make sure 2-Step Verification is on |
| NewsAPI returns no articles | Free tier has some query limits; try simplifying company names |
| Claude API error | Check your `ANTHROPIC_API_KEY` and account credit balance |

All runs are logged to `output/agent.log`.

---

## How it works

```
data/portfolio.csv
     │
     ▼
yfinance ──→ live prices + weekly/monthly/YTD changes, all in EUR (daily FX)
     │
     ├──→ S&P 500 / Nasdaq / MSCI World benchmarks, also in EUR
     ├──→ earnings + ex-dividend dates for the next 14 days
     │
NewsAPI ──→ articles → Haiku relevance filter (0–2 score) → top 10 relevant + world digest
     │
     ▼
Claude (claude-sonnet-5, override with CLAUDE_MODEL)
  ├── Portfolio Performance Summary
  ├── Benchmark Comparison
  ├── News & Market Context
  └── Actionable Suggestions
     │
     ▼
HTML email → Gmail SMTP → your inbox
```

The email also includes, when there is data for them: **Allocation vs Target** (see step 8), **Coming Up** (earnings and ex-dividend dates), and a **Track Record** of earlier recommendations — ADD counts as a hit when the price rose since, TRIM/EXIT when it fell.

A red banner under the masthead warns when the holdings export is more than 30 days old, or when the import left out holdings whose ISIN could not be mapped to a ticker (they are listed in the notes at the end).

The history chart compares a deposit-free portfolio index with the S&P 500 in EUR. Each weekly run chains the price return of your current holdings since the previous run, so money you pay in or take out never shows up as performance. The chart appears once four weekly data points with the index exist.



---

## 7 — Update your holdings (Finanzfluss users)

If you track your portfolio in [Finanzfluss](https://finanzfluss.de), you can use the included import script to keep `data/portfolio.csv` up to date.

### Export from Finanzfluss

1. Open Finanzfluss → **Portfolio** → **Depot**
2. Click **Export** → save the CSV as `data/finanzfluss_export.csv`

### Run the import

```bash
python import_portfolio.py --input data/finanzfluss_export.csv
```

The script resolves ISINs to Yahoo Finance tickers automatically, in this order: your `data/isin_map.json` overrides → [OpenFIGI](https://www.openfigi.com) (free bulk API) → yfinance search (slow fallback). **`isin_map.json` is optional** — most holdings resolve automatically via OpenFIGI without it. It's only needed for Finanzfluss imports, and only as a correction layer for the exceptions: ISINs OpenFIGI can't resolve, or ETFs where it picks the wrong exchange listing (e.g. London instead of Xetra). If you edit `portfolio.csv` manually instead (step 4), you'll never need this file at all.

If you hit one of those exceptions, add an override:

```bash
cp data/isin_map.json.example data/isin_map.json
# then edit data/isin_map.json and add your ISINs
```

> **Rate limits:** OpenFIGI allows 25 requests/min without a key, 250/min with one. With more than a couple hundred holdings you may hit the free limit — if so, get a free key at [openfigi.com](https://www.openfigi.com) and add `OPENFIGI_API_KEY=...` to your `.env`. Not needed otherwise.

> **Note:** everything in `data/` except the `*.example` templates is in `.gitignore` — your holdings stay private.

---

## 8 — Target allocation (optional)

Set target weights for groups of holdings and the report shows how far each group has drifted:

```bash
cp data/targets.json.example data/targets.json
# then edit groups, target_pct and tickers (Yahoo tickers as in data/portfolio.csv)
```

Holdings in no group are shown as *Unassigned*. Groups that drift by more than `tolerance_pct` percentage points are highlighted and fed to Claude for the actionable suggestions. `targets.json` is gitignored; for GitHub Actions, upload it to your private data repo next to `isin_map.json`.

---

## 9 — Paid in vs market gain (optional)

The masthead can split the value change into money you paid in and actual market gain. Export your account transactions from Finanzfluss as `data/finanzfluss_transactions.csv`, then:

```bash
python import_cashflows.py --input data/finanzfluss_transactions.csv
```

This writes `data/cashflows.csv` with only daily totals of money invested (buys minus sells) and dividends/interest — private spending, counterparties and transfers are dropped. Upload `cashflows.csv` (never the raw export) to your private data repo for GitHub Actions. The split runs from the first history entry of the current year, and the email warns when the cash-flow data is more than 30 days old. Both files are gitignored.

The same data drives the **Total** figure in the masthead: current value minus everything paid in since the first booking, so it includes realised gains and losses of sold positions. This only holds if every purchase ran through that one cash account and the export covers its full history — positions bought through another broker or moved in by a securities transfer (Depotübertrag) would show up as gains. Without `cashflows.csv` the Total figure is simply left out.


## 10 — Interactive dashboard (optional)

Next to the email, you can build a performance dashboard for your browser:

```bash
python dashboard.py --open
```

This writes `output/dashboard.html`, a single file you open locally (it loads Plotly from a CDN, so it needs an internet connection to display). It fetches prices only — no Claude, no news, no email — and takes about half a minute. Sections:

- **Your actual record** — net paid in vs value, total gain and your money-weighted return p.a. (XIRR), from `cashflows.csv` and `history.csv` (section 9).
- **Backtest of today's portfolio** — today's share counts held unchanged over up to 7 years against the S&P 500 and MSCI World (total return, in EUR): range buttons, drawdown, returns by period, monthly heatmap, risk table (Sharpe, Sortino, beta, tracking error, capture ratios, VaR…) and rolling metrics. This shows how your current holdings behave, not your actual past performance.
- **Positions** — sortable table with return and contribution per period, volatility, beta, risk share and momentum.
- **Structure** — concentration, diversification ratio, sectors, regions, weight vs risk share and a correlation heatmap.

The file contains your holdings and euro amounts. Keep it local, or host it only behind a login.
