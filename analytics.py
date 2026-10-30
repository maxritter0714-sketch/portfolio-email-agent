"""Performance and risk analytics for the dashboard. Pure functions, no I/O.

Return series are daily simple returns on a shared calendar. "Portfolio"
means today's holdings with today's share counts, held unchanged over the
whole history (a backtest of the current portfolio, not the real track
record); the real record comes from cash flows (see `xirr`).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

TDAYS = 252
WEEKS = 52

PERIODS = ("1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y")
_PERIOD_OFFSETS = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "3Y": pd.DateOffset(years=3),
    "5Y": pd.DateOffset(years=5),
}


# ── Portfolio series ─────────────────────────────────────────────────────────

def buy_hold_returns(prices: pd.DataFrame, shares: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Daily return of holding today's share counts, plus listing coverage.

    `prices` are EUR closes (dates × tickers). Gaps (holidays) are carried
    forward. A name only counts from the day after its first price, so a
    later listing extends the portfolio without creating a jump. Coverage is
    the share of today's value whose name was already listed on that day.
    """
    px = prices.sort_index().ffill()
    sh = shares.reindex(px.columns).fillna(0.0)
    values = px.mul(sh, axis=1)
    prev = values.shift(1)
    both = values.notna() & prev.notna()
    gain = (values - prev).where(both).sum(axis=1)
    base = prev.where(both).sum(axis=1)
    ret = (gain / base.replace(0.0, np.nan)).iloc[1:].dropna()

    today = values.iloc[-1].fillna(0.0)
    total = today.sum()
    weights = today / total if total else today
    coverage = values.notna().mul(weights, axis=1).sum(axis=1)
    return ret, coverage


def index_from_returns(ret: pd.Series, base: float = 100.0) -> pd.Series:
    return (1.0 + ret).cumprod() * base


def to_weekly(ret: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compound daily returns into weeks ending Friday.

    Co-movement measures (beta, correlation, tracking error, covariance) use
    weekly returns: Xetra closes hours before New York, so on daily data a
    US move shows up a day late in EUR listings and correlations come out
    far too low (an S&P 500 ETF on Xetra shows a daily beta of ~0.5).
    """
    return (1.0 + ret).resample("W-FRI").prod(min_count=1) - 1.0


def drawdown_series(index: pd.Series) -> pd.Series:
    """Distance below the running peak, 0 at a new high (e.g. -0.12 = 12 % under)."""
    return index / index.cummax() - 1.0


def drawdown_stats(index: pd.Series) -> dict:
    """Max and current drawdown plus the longest stretch spent under a prior peak."""
    if index.empty:
        return dict(max_dd=None, max_dd_date=None, current_dd=None, longest_days=None)
    dd = drawdown_series(index)
    longest = 0
    start = None
    for ts, value in dd.items():
        if value < 0 and start is None:
            start = ts
        elif value >= 0 and start is not None:
            longest = max(longest, (ts - start).days)
            start = None
    if start is not None:
        longest = max(longest, (dd.index[-1] - start).days)
    return dict(
        max_dd=float(dd.min()),
        max_dd_date=dd.idxmin().date().isoformat(),
        current_dd=float(dd.iloc[-1]),
        longest_days=longest,
    )


# ── Period returns ───────────────────────────────────────────────────────────

def period_start(end: pd.Timestamp, key: str) -> pd.Timestamp:
    """Reference date for a period: the YTD base is the last close of the prior year."""
    if key == "YTD":
        return pd.Timestamp(end.year - 1, 12, 31)
    return end - _PERIOD_OFFSETS[key]


def period_return(index: pd.Series, start: pd.Timestamp) -> float | None:
    """Return from the last value at or before `start` to the end; None if history is too short."""
    before = index[index.index <= start]
    if before.empty:
        return None
    return float(index.iloc[-1] / before.iloc[-1] - 1.0)


def cagr(total_return: float, days: int) -> float | None:
    if days <= 0 or total_return <= -1:
        return None
    return float((1.0 + total_return) ** (365.25 / days) - 1.0)


def period_table(indices: dict[str, pd.Series], end: pd.Timestamp) -> list[dict]:
    """Return per period for each named index; CAGR added for periods of a year or more."""
    rows = []
    for key in PERIODS:
        start = period_start(end, key)
        row: dict = dict(key=key, start=start.date().isoformat())
        for name, idx in indices.items():
            r = period_return(idx, start)
            row[name] = r
            if key in ("1Y", "3Y", "5Y") and r is not None:
                row[f"{name}_cagr"] = cagr(r, (end - start).days)
        rows.append(row)
    return rows


def monthly_returns(ret: pd.Series) -> dict:
    """Year × month grid of compounded returns plus a per-year total."""
    monthly = (1.0 + ret).resample("ME").prod() - 1.0
    years = sorted({ts.year for ts in monthly.index})
    grid = {y: [None] * 12 for y in years}
    for ts, value in monthly.items():
        grid[ts.year][ts.month - 1] = float(value)
    totals = {
        y: float(np.prod([1.0 + v for v in grid[y] if v is not None]) - 1.0)
        for y in years
    }
    return dict(years=years, grid=[grid[y] for y in years], totals=[totals[y] for y in years])


# ── Risk ─────────────────────────────────────────────────────────────────────

def _monthly(ret: pd.Series) -> pd.Series:
    return (1.0 + ret).resample("ME").prod() - 1.0


def capture_ratios(port: pd.Series, bench: pd.Series) -> tuple[float | None, float | None]:
    """Up/down capture on monthly returns: share of the benchmark's average up / down month.

    Up 110 % means the portfolio gained 1.1× the benchmark in its rising
    months; down 90 % means it lost 0.9× as much in falling months.
    """
    pm, bm = _monthly(port), _monthly(bench)
    pm, bm = pm.align(bm, join="inner")

    def ratio(mask: pd.Series) -> float | None:
        if mask.sum() == 0:
            return None
        p = np.prod(1.0 + pm[mask]) ** (1.0 / mask.sum()) - 1.0
        b = np.prod(1.0 + bm[mask]) ** (1.0 / mask.sum()) - 1.0
        return float(p / b) if b else None

    return ratio(bm > 0), ratio(bm < 0)


def risk_stats(port: pd.Series, sp500: pd.Series, world: pd.Series, risk_free: float) -> dict:
    """Return and risk figures for one window of aligned daily returns.

    Beta, alpha and correlation are measured against the S&P 500 (as in the
    email); tracking error, information ratio and capture against the MSCI
    World, the passive alternative to the portfolio. Co-movement figures use
    weekly returns (see `to_weekly`).
    """
    df = pd.concat({"p": port, "s": sp500, "w": world}, axis=1).dropna()
    if len(df) < 60:
        return {}
    p, s, w = df["p"].to_numpy(), df["s"].to_numpy(), df["w"].to_numpy()
    wk = to_weekly(df).dropna()
    pw, sw, ww = wk["p"].to_numpy(), wk["s"].to_numpy(), wk["w"].to_numpy()
    days = (df.index[-1] - df.index[0]).days
    total = float(np.prod(1.0 + p) - 1.0)
    ann_ret = cagr(total, days)
    world_ann = cagr(float(np.prod(1.0 + w) - 1.0), days)

    mean_ann = float(np.mean(p) * TDAYS)
    vol = float(np.std(p, ddof=1) * np.sqrt(TDAYS))
    downside = float(np.sqrt(np.mean(np.minimum(p, 0.0) ** 2)) * np.sqrt(TDAYS))
    cov = np.cov(pw, sw)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] else None
    alpha = (
        mean_ann - (risk_free + beta * (float(np.mean(s) * TDAYS) - risk_free))
        if beta is not None else None
    )
    te = float(np.std(pw - ww, ddof=1) * np.sqrt(WEEKS))
    dd = drawdown_stats(index_from_returns(df["p"]))
    up, down = capture_ratios(df["p"], df["w"])
    months = _monthly(df["p"])
    var_95 = float(np.percentile(p, 5))

    return dict(
        start=df.index[0].date().isoformat(),
        cagr=ann_ret,
        volatility=vol,
        sharpe=(mean_ann - risk_free) / vol if vol else None,
        sortino=(mean_ann - risk_free) / downside if downside else None,
        max_dd=dd["max_dd"],
        calmar=ann_ret / abs(dd["max_dd"]) if ann_ret is not None and dd["max_dd"] else None,
        longest_dd_days=dd["longest_days"],
        beta=beta,
        alpha=alpha,
        correlation=float(np.corrcoef(pw, sw)[0, 1]),
        tracking_error=te,
        information_ratio=(ann_ret - world_ann) / te if te and ann_ret is not None and world_ann is not None else None,
        up_capture=up,
        down_capture=down,
        var_95=var_95,
        cvar_95=float(np.mean(p[p <= var_95])),
        best_month=float(months.max()),
        worst_month=float(months.min()),
        positive_months=float((months > 0).mean()),
    )


def rolling_stats(port: pd.Series, sp500: pd.Series, risk_free: float) -> dict[str, pd.Series]:
    """Rolling 90d Sharpe, 30d volatility, 26-week beta and trailing 1-year return."""
    df = pd.concat({"p": port, "s": sp500}, axis=1).dropna()
    p = df["p"]
    mean90 = p.rolling(90).mean() * TDAYS
    std90 = p.rolling(90).std() * np.sqrt(TDAYS)
    wk = to_weekly(df).dropna()
    return dict(
        return_1y=np.expm1(np.log1p(p).rolling(TDAYS).sum()).dropna(),
        sharpe_90d=((mean90 - risk_free) / std90).dropna(),
        vol_30d=(p.rolling(30).std() * np.sqrt(TDAYS)).dropna(),
        beta_26w=(wk["p"].rolling(26).cov(wk["s"]) / wk["s"].rolling(26).var()).dropna(),
    )


# ── Per position and structure ──────────────────────────────────────────────

def contributions(prices: pd.DataFrame, shares: pd.Series, start: pd.Timestamp) -> pd.DataFrame:
    """Per-name return and contribution since `start`, holding today's share counts.

    Contribution is the name's EUR gain over the portfolio's start value, so
    contributions add up to the portfolio return over names listed at `start`.
    """
    px = prices.sort_index().ffill()
    before = px[px.index <= start]
    if before.empty:
        return pd.DataFrame(columns=["ret", "contrib"])
    p0, p1 = before.iloc[-1], px.iloc[-1]
    sh = shares.reindex(px.columns).fillna(0.0)
    v0 = (p0 * sh).dropna()
    base = v0.sum()
    ret = p1 / p0 - 1.0
    contrib = (sh * (p1 - p0)) / base if base else ret * np.nan
    return pd.DataFrame({"ret": ret, "contrib": contrib})


def position_risk(returns: pd.DataFrame, port: pd.Series, sp500: pd.Series) -> pd.DataFrame:
    """Per-name annualised volatility, beta to the S&P 500, max drawdown and correlation to the portfolio.

    Beta and correlation use weekly returns (see `to_weekly`).
    """
    rows = {}
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        if len(r) < 60:
            continue
        df = to_weekly(pd.concat({"r": r, "s": sp500, "p": port}, axis=1).dropna()).dropna()
        var_s = df["s"].var()
        rows[ticker] = dict(
            volatility=float(r.std() * np.sqrt(TDAYS)),
            beta=float(df["r"].cov(df["s"]) / var_s) if var_s else None,
            max_dd=float(drawdown_series(index_from_returns(r)).min()),
            corr_portfolio=float(df["r"].corr(df["p"])),
        )
    return pd.DataFrame.from_dict(rows, orient="index")


def risk_contributions(returns: pd.DataFrame, weights: pd.Series) -> tuple[pd.Series, float, float]:
    """Each name's share of portfolio variance, plus portfolio volatility and diversification ratio.

    The diversification ratio is the weighted average of the names' own
    volatilities over the portfolio volatility: 1.0 means no diversification
    benefit, higher means the names offset each other. `returns` are daily;
    the covariance is estimated on weekly returns (see `to_weekly`).
    """
    cols = [c for c in returns.columns if returns[c].notna().sum() >= 60 and c in weights.index]
    if not cols:
        return pd.Series(dtype=float), 0.0, 0.0
    w = weights.reindex(cols).fillna(0.0)
    w = w / w.sum()
    cov = to_weekly(returns[cols]).cov().fillna(0.0) * WEEKS
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return pd.Series(0.0, index=cols), 0.0, 0.0
    share = w * (cov @ w) / port_var
    vol = np.sqrt(port_var)
    own_vol = np.sqrt(np.diag(cov))
    return share, float(vol), float((w.to_numpy() @ own_vol) / vol)


def concentration(weights: pd.Series) -> dict:
    """HHI, effective number of positions and top-5/top-10 weight share."""
    w = weights.sort_values(ascending=False)
    hhi = float((w ** 2).sum())
    return dict(
        hhi=hhi,
        effective_n=1.0 / hhi if hhi else None,
        top5=float(w.head(5).sum()),
        top10=float(w.head(10).sum()),
        count=int(len(w)),
    )


# ── Real track record ───────────────────────────────────────────────────────

def xirr(flows: list[tuple[date, float]]) -> float | None:
    """Annualised money-weighted return of dated cash flows (money paid in negative).

    Solved by bisection; None when the flows have no sign change or no root
    between -99 % and +1000 % a year.
    """
    flows = [(d, a) for d, a in flows if a]
    if not flows or min(a for _, a in flows) >= 0 or max(a for _, a in flows) <= 0:
        return None
    d0 = min(d for d, _ in flows)
    years = np.array([(d - d0).days / 365.25 for d, _ in flows])
    amounts = np.array([a for _, a in flows])

    def npv(rate: float) -> float:
        return float(np.sum(amounts / (1.0 + rate) ** years))

    lo, hi = -0.99, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-6:
            break
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return float((lo + hi) / 2)


def portfolio_xirr(cashflows: list[dict], current_value: float, today: date) -> float | None:
    """XIRR of buys/sells and income from cashflows.csv, closed out at today's value."""
    flows = [(date.fromisoformat(c["date"]), -c["invested_eur"]) for c in cashflows]
    flows += [(date.fromisoformat(c["date"]), c["income_eur"]) for c in cashflows]
    flows.append((today, current_value))
    return xirr(flows)


def cumulative_flows(cashflows: list[dict]) -> list[dict]:
    """Running totals of net money invested and income received, one row per booking day."""
    invested = income = 0.0
    out = []
    for c in cashflows:
        invested += c["invested_eur"]
        income += c["income_eur"]
        out.append(dict(date=c["date"], invested=invested, income=income))
    return out


def counterfactual_value(cashflows: list[dict], prices: pd.Series) -> pd.Series:
    """Value over time had every net euro paid in bought `prices` (e.g. an index fund) instead.

    Each booking buys (or, when negative, sells) units at the first close on or
    after its date; income is left out, as the index price already reinvests it.
    """
    prices = prices.dropna()
    if prices.empty or not cashflows:
        return pd.Series(dtype=float)
    units = pd.Series(0.0, index=prices.index)
    for c in cashflows:
        pos = min(prices.index.searchsorted(pd.Timestamp(c["date"])), len(prices) - 1)
        units.iloc[pos] += c["invested_eur"] / prices.iloc[pos]
    first = pd.Timestamp(cashflows[0]["date"])
    held = units.cumsum() * prices
    return held[held.index >= min(first, prices.index[-1])]


def monthly_flows(cashflows: list[dict]) -> list[dict]:
    """Net money invested and income received per calendar month."""
    if not cashflows:
        return []
    frame = pd.DataFrame(cashflows)
    frame["month"] = frame["date"].str[:7]
    sums = frame.groupby("month")[["invested_eur", "income_eur"]].sum()
    return [dict(month=m, invested=float(r.invested_eur), income=float(r.income_eur))
            for m, r in sums.iterrows()]


# ── Recommendation track record ─────────────────────────────────────────────

# ADD should beat the benchmark afterwards; TRIM and EXIT should lag it. RESEARCH has no direction.
_REC_DIRECTION = {"ADD": 1, "TRIM": -1, "EXIT": -1}


def _close_on(prices: pd.Series, day: pd.Timestamp) -> float | None:
    before = prices.dropna()
    if before.empty:
        return None
    before = before[before.index <= day]
    return float(before.iloc[-1]) if not before.empty else None


def recommendation_record(recs: dict[str, dict], prices: pd.DataFrame, bench: pd.Series) -> list[dict]:
    """Return of each recommended name since the recommendation, against the benchmark.

    A hit is judged on the return relative to the benchmark (`excess`), not the
    raw price, so a falling market doesn't make every TRIM look clever.
    """
    rows = []
    for ticker, rec in recs.items():
        day = pd.Timestamp(rec["date"])
        series = prices[ticker].dropna() if ticker in prices.columns else pd.Series(dtype=float)
        then, now = _close_on(series, day), (float(series.iloc[-1]) if not series.empty else None)
        b_then, b_now = _close_on(bench, day), _close_on(bench, bench.index[-1])
        ret = now / then - 1 if then and now else None
        bench_ret = b_now / b_then - 1 if b_then and b_now else None
        excess = ret - bench_ret if ret is not None and bench_ret is not None else None
        direction = _REC_DIRECTION.get(rec.get("action", ""))
        hit = None if direction is None or excess is None else (excess > 0) == (direction > 0)
        rows.append(dict(ticker=ticker, action=rec.get("action", ""), report=rec.get("report"),
                         date=rec["date"], ret=ret, bench_ret=bench_ret, excess=excess, hit=hit))
    rows.sort(key=lambda r: (r["date"], r["ticker"]), reverse=True)
    return rows


# ── Simulations ─────────────────────────────────────────────────────────────

def window_returns(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return of each column from the last close on or before `start` to the last on or before `end`.

    NaN for names that weren't listed yet at `start`.
    """
    filled = prices.ffill()
    before = filled[filled.index <= start]
    until = filled[filled.index <= end]
    if before.empty or until.empty:
        return pd.Series(np.nan, index=prices.columns)
    return until.iloc[-1] / before.iloc[-1] - 1


def stress_test(prices: pd.DataFrame, weights: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                fill: float) -> tuple[float, pd.DataFrame]:
    """Today's weights through a past episode, held without trading.

    Names not yet listed at `start` get `fill` (the benchmark's return) and are flagged.
    Returns the portfolio return and a frame of ret / contrib / filled per name.
    """
    ret = window_returns(prices.reindex(columns=weights.index), start, end)
    filled = ret.isna()
    ret = ret.fillna(fill)
    contrib = weights * ret
    return float(contrib.sum()), pd.DataFrame({"ret": ret, "contrib": contrib, "filled": filled})


def average_monthly_savings(cashflows: list[dict], today: date, months: int = 12) -> float | None:
    """Mean net money paid in per month over the last `months` full months (empty months count as 0)."""
    if not cashflows:
        return None
    this_month = pd.Period(today, "M")
    window = {str(this_month - i) for i in range(1, months + 1)}
    total = sum(m["invested"] for m in monthly_flows(cashflows) if m["month"] in window)
    return total / months
