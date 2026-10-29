# portfolioAgent — CLAUDE.md

## Project
**Portfolio Email Agent** — sends an HTML portfolio report every other Saturday.
Stack: Python · yfinance · NewsAPI · Anthropic Claude · Gmail SMTP

## Key files
| File | Purpose |
|---|---|
| `agent.py` | CLI: schedule, orchestration, Gmail send |
| `market.py` | Portfolio load, FX, prices, history, charts, send cadence |
| `metrics.py` | Sharpe/beta/momentum/look-through |
| `news.py` | NewsAPI + Haiku relevance filter |
| `claude_report.py` | Claude analysis, headline, conviction |
| `email_html.py` | HTML email builder |
| `theme.py` | Shared colours for charts and email |
| `import_portfolio.py` | Converts Finanzfluss export CSV → `data/portfolio.csv` |
| `import_cashflows.py` | Reduces the Finanzfluss transactions export → daily `data/cashflows.csv` (only invested + dividends; private spending stays local) |
| `paths.py` | `DATA_DIR` (`data/`) and `OUTPUT_DIR` (`output/`) — all file locations hang off these |
| `data/` | All personal files, gitignored except `*.example`: `finanzfluss_export.csv`, `portfolio.csv`, `history.csv`, `conviction.json`, `cashflows.csv`, `isin_map.json`, `targets.json` |
| `output/` | `agent.log`, `preview.html` (gitignored) |
| `.env` | API keys and Gmail credentials (never commit) |
| `requirements.txt` | Python dependencies |

## Workflow — updating holdings
1. Export from Finanzfluss → save as `data/finanzfluss_export.csv`
2. Add any new ISIN overrides to `data/isin_map.json` (copy from `data/isin_map.json.example` if it doesn't exist yet — gitignored, never commit)
3. Run: `python import_portfolio.py --input data/finanzfluss_export.csv`
4. Verify `data/portfolio.csv` looks correct
5. Optional: export account transactions as `data/finanzfluss_transactions.csv` (gitignored, contains private spending — never upload it), run `python import_cashflows.py --input data/finanzfluss_transactions.csv`, upload only `data/cashflows.csv` to the data repo (the data repo itself stays flat; the workflow copies its files into `data/`)
6. Test: `python agent.py --force --preview`

## ISIN map rules
- Resolution priority: `data/isin_map.json` overrides (gitignored, personal) → OpenFIGI bulk API (Bloomberg, free, 100 ISINs per call) → yfinance Search API (slow, unreliable — last resort)
- `isin_map.json` exists to correct cases where OpenFIGI picks the wrong exchange listing (e.g. London vs. Xetra). Irish and Luxembourg UCITS ETFs prefer Xetra (`GY`) over London (`LN`) so Yahoo tickers match the EUR listing you actually hold.
- For the cloud (GitHub Actions) run, overrides are pulled from the private `portfolio-agent-data` repo (`_data/isin_map.json`) — keep both copies in sync when you edit one
- Zertifikate/OS positions are always excluded
- Positions with `Anzahl = 0` are skipped (sold/zero positions kept for history)

## Commit style
Conventional Commits: `type(scope): description`
Examples: `feat(import): add Fluence Energy ISIN`, `fix(agent): handle missing ticker`

## Session logs
Stored in `.claude/sessions/YYYY-MM-DD-slug.md`
Run `/session-start` at the beginning of each session.
Run `/session-end` at the end to write the handoff log.



