# portfolioAgent — AGENTS.md

This project is developed primarily with Claude Code. **`CLAUDE.md` is the
single source of truth** for project context, key files, the holdings
workflow, ISIN map rules, and commit style — read it first. This file only
adds notes for other agents, mostly used for code review.

## Review notes
- Python 3.11. Run tests with `pytest -q` (same as CI, `.github/workflows/test.yml`).
- Tests are pure-function tests without HTTP mocking — don't flag missing
  network mocks as a gap.
- Architectural decisions live in `.claude/decisions/` (ADRs, gitignored —
  only available in a local checkout). Check them before suggesting changes
  to settled design choices (e.g. fortnight send cadence, ISIN resolution,
  single-call Claude analysis).
- Never read out, print, or commit secrets or personal data: `.env`,
  `portfolio.csv`, `finanzfluss_export.csv`, `isin_map.json`, `history.csv`,
  `conviction.json`.
- Reviews are read-only: report findings, don't commit or push.
