# Portfolio Email Agent

Sends you a polished HTML portfolio report every **second Saturday of the month**.  
The report includes live prices, P&L, benchmark comparison, curated news, and a four-section AI analysis from Claude.

> **Heads up:** This project was built for personal use and may require some adaptation before it works out of the box for you. In particular:
> - **ISIN → ticker overrides** live in `isin_map.json` (gitignored, not included). Copy `isin_map.json.example` to get started — most ISINs are resolved automatically via OpenFIGI
> - The **sector and region classification** in `agent.py` is resolved dynamically from yfinance — no hardcoded maps to maintain
> - The **Finanzfluss import script** expects column names from a Finanzfluss export — other brokers will need column name adjustments
> - The agent is built around **EUR as base currency**

---

## Prerequisites

- Python 3.10 or newer
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

Edit `portfolio.csv` with your real holdings:

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

Run with `--force` to bypass the second-Saturday check:

```bash
python agent.py --force
```

You should receive an email within about 30 seconds. Check `agent.log` if something goes wrong.

---

## 6 — Schedule it

### GitHub Actions (recommended)

The cleanest setup — runs in the cloud, no local machine required, secrets stored encrypted.

1. Create a private GitHub repo (e.g. `portfolio-agent-data`) and upload your `finanzfluss_export.csv`, `history.csv`, and `conviction.json` there
2. Create a fine-grained Personal Access Token scoped to that private repo with **Contents: Read and write**
3. Add secrets to your public repo under **Settings → Secrets and variables → Actions**:
   - `DATA_REPO_TOKEN` — the PAT from step 2
   - `ANTHROPIC_API_KEY`, `NEWS_API_KEY`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`, `RECIPIENT_EMAIL`
4. The included `.github/workflows/report.yml` runs every Saturday at 6am UTC automatically

The workflow collects a weekly data point on every Saturday and sends the full email report on second Saturdays only. `history.csv` and `conviction.json` are pushed back to the private repo after each run so they persist between executions.

To trigger a test run: **Actions → Portfolio Report → Run workflow → tick Force full report**.

### Mac / Linux (cron)

Open your crontab:

```bash
crontab -e
```

Add this line to run at 08:00 on every Saturday (the script self-checks for the second Saturday):

```cron
0 8 * * 6 cd /path/to/portfolioAgent && /usr/bin/python3 agent.py >> agent.log 2>&1
```

Replace `/path/to/portfolioAgent` with the full path to this directory and `/usr/bin/python3` with the output of `which python3`.

For weekly data collection without sending an email:

```cron
0 8 * * 6 cd /path/to/portfolioAgent && /usr/bin/python3 agent.py --data-only >> agent.log 2>&1
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `portfolio.csv not found` | Make sure you're running the script from the `portfolioAgent` directory |
| `No price history returned` | Check the ticker symbol on [finance.yahoo.com](https://finance.yahoo.com) |
| Gmail authentication error | Re-generate the App Password; make sure 2-Step Verification is on |
| NewsAPI returns no articles | Free tier has some query limits; try simplifying company names |
| Claude API error | Check your `ANTHROPIC_API_KEY` and account credit balance |

All runs are logged to `agent.log` in the project directory.

---

## How it works

```
portfolio.csv
     │
     ▼
yfinance ──→ live prices + weekly/monthly/YTD changes (USD→EUR conversion)
     │
     ├──→ S&P 500 / Nasdaq / MSCI World benchmark comparison
     │
NewsAPI ──→ 30 articles → Haiku relevance filter (0–2 score) → top 10 relevant + 5 general
     │
     ▼
Claude (claude-sonnet-4-6)
  ├── Portfolio Performance Summary
  ├── Benchmark Comparison
  ├── News & Market Context
  └── Actionable Suggestions
     │
     ▼
HTML email → Gmail SMTP → your inbox
```



---

## 7 — Update your holdings (Finanzfluss users)

If you track your portfolio in [Finanzfluss](https://finanzfluss.de), you can use the included import script to keep `portfolio.csv` up to date.

### Export from Finanzfluss

1. Open Finanzfluss → **Portfolio** → **Depot**
2. Click **Export** → download the CSV

### Run the import

```bash
python import_portfolio.py --input finanzfluss_export.csv
```

The script resolves ISINs to Yahoo Finance tickers via OpenFIGI and writes `portfolio.csv`.

If any ISINs can't be resolved, or if OpenFIGI picks the wrong exchange listing for an ETF, add an override to `isin_map.json`:

```bash
cp isin_map.json.example isin_map.json
# then edit isin_map.json and add your ISINs
```

> **Note:** `finanzfluss_export.csv` and `isin_map.json` are in `.gitignore` — your holdings stay private.




