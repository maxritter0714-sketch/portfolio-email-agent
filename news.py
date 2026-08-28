"""NewsAPI fetch, ticker tagging, and Haiku relevance filter."""
from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from difflib import SequenceMatcher
from typing import Any

import anthropic
from newsapi import NewsApiClient

log = logging.getLogger(__name__)

_NAME_STOP = {
    "inc", "inc.", "corp", "corp.", "corporation", "ltd", "ltd.", "plc",
    "group", "holdings", "holding", "co", "co.", "nv", "n.v.", "sa", "s.a.",
    "se", "ag", "ordinary", "registered", "shares", "the", "and", "of",
    "com", "technologies", "technology", "systems", "properties",
}

# First-token tags skipped: these are English/finance words as well as company names.
_GENERIC_NAME_TOKENS = frozenset({
    "apple", "meta", "visa", "general", "target", "super", "trade",
    "advanced", "shell", "discovery", "united", "analog", "pacific",
    "national", "digital", "global", "energy", "capital", "international",
    "american", "partner", "partners", "public", "social", "match",
    "block", "square", "snow", "core", "plus", "next", "first", "real",
    "open", "smart", "best", "city", "world", "data", "tech", "micro",
})

_NEWS_QUERY_LIMIT = 500
_NEWS_QUERY_SUFFIX = " AND (stock OR earnings OR revenue OR market)"
_NEWS_ARTICLE_CAP = 40


def _extract_text_from_response(response: Any) -> str:
    content = getattr(response, "content", None)
    if not content:
        return ""
    for block in content:
        if isinstance(block, str):
            return block
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
    return ""


def _first_token(name: str) -> str:
    """Return the first significant word in a company name, lowercased."""
    # Drop everything after a parenthesis
    clean = re.split(r"[(\-]", name)[0]
    words = re.split(r"[\s,\.]+", clean.lower())
    words = [w for w in words if w and w not in _NAME_STOP]
    return words[0] if words else ""


def _word_in(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return re.search(r"\b" + re.escape(needle) + r"\b", haystack, re.IGNORECASE) is not None


def _clean_query_name(name: str) -> str:
    return name.replace("&", "and").replace("(", "").replace(")", "").replace('"', "").strip()


def _query_name_batches(names: list[str], max_name_chars: int) -> list[str]:
    """OR-joined quoted names, each clause no longer than max_name_chars."""
    batches: list[str] = []
    parts: list[str] = []
    used = 0
    for name in names:
        clean = _clean_query_name(name)
        if not clean:
            continue
        token = f'"{clean}"'
        if len(token) > max_name_chars:
            log.warning("Skipping oversized news query name (%d chars)", len(token))
            continue
        joiner = len(" OR ") if parts else 0
        if used + joiner + len(token) > max_name_chars:
            batches.append(" OR ".join(parts))
            parts = [token]
            used = len(token)
            continue
        parts.append(token)
        used += joiner + len(token)
    if parts:
        batches.append(" OR ".join(parts))
    return batches


def _ticker_mentioned(raw: str, ticker: str) -> bool:
    """True if the article cites this ticker, not an English word like 'meta'."""
    root = ticker.split(".", 1)[0]
    if root.lower() in _GENERIC_NAME_TOKENS:
        return re.search(r"\b" + re.escape(ticker) + r"\b", raw) is not None
    if _word_in(raw, ticker):
        return True
    if "." in ticker and len(root) >= 4 and _word_in(raw, root):
        return True
    return False


def _name_phrases(name: str) -> list[str]:
    """Search strings for a holding: two-word phrase, then distinctive first token."""
    clean = re.split(r"[(\-]", name)[0]
    words = [w for w in re.split(r"[\s,\.]+", clean.lower()) if w and w not in _NAME_STOP]
    phrases: list[str] = []
    if len(words) >= 2:
        phrases.append(" ".join(words[:2]))
    if words and len(words[0]) >= 4 and words[0] not in _GENERIC_NAME_TOKENS:
        phrases.append(words[0])
    return phrases


def _tag_article_tickers(article: dict, positions: list[dict]) -> list[str]:
    """Return tickers from positions mentioned in the article."""
    raw = (article.get("headline", "") + " " + article.get("description", ""))
    text = raw.lower()
    matched: list[str] = []
    for p in positions:
        ticker = p["ticker"]
        name = p["name"]
        if _ticker_mentioned(raw, ticker):
            matched.append(ticker)
            continue
        full = re.split(r"[(\-]", name)[0].strip().lower()
        if len(full) >= 5 and full in text:
            matched.append(ticker)
            continue
        if any(_word_in(text, phrase) for phrase in _name_phrases(name)):
            matched.append(ticker)
    seen: set[str] = set()
    out: list[str] = []
    for t in matched:
        if t not in seen:
            seen.add(t)
            out.append(t)
            if len(out) >= 3:
                break
    return out


def fetch_portfolio_news(positions: list[dict]) -> list[dict]:
    """Fetch recent articles for every holding (batched to NewsAPI's query limit)."""
    api_key = os.getenv("NEWS_API_KEY", "")
    newsapi = NewsApiClient(api_key=api_key)

    names = [p["name"] for p in positions]
    max_name_chars = _NEWS_QUERY_LIMIT - len("(") - len(")") - len(_NEWS_QUERY_SUFFIX)
    batches = _query_name_batches(names, max_name_chars)
    if not batches:
        log.warning("No usable position names for portfolio news query -- skipping")
        return []

    raw: list[dict] = []
    for i, name_clause in enumerate(batches, 1):
        query = f"({name_clause}){_NEWS_QUERY_SUFFIX}"
        try:
            resp = newsapi.get_everything(
                q=query, language="en", sort_by="relevancy", page_size=20,
            )
            batch = resp.get("articles", [])
            raw.extend(batch)
            log.info(
                "Portfolio news batch %d/%d: %d hits",
                i, len(batches), len(batch),
            )
        except Exception as exc:
            log.error("Portfolio news batch %d/%d failed: %s", i, len(batches), exc)

    out: list[dict] = []
    seen: set[str] = set()
    for a in raw:
        title = (a.get("title") or "").strip()
        if not title or title == "[Removed]" or title in seen:
            continue
        seen.add(title)
        article = dict(
            headline=title,
            description=a.get("description") or "",
            source=a.get("source", {}).get("name", "Unknown"),
            date=(a.get("publishedAt") or "")[:10],
            url=a.get("url", ""),
        )
        article["tickers"] = _tag_article_tickers(article, positions)
        out.append(article)
        if len(out) >= _NEWS_ARTICLE_CAP:
            break
    log.info("Fetched %d portfolio news articles (%d query batches)", len(out), len(batches))
    return out


def filter_news_by_relevance(
    articles: list[dict],
    positions: list[dict],
    min_score: int = 2,
    max_items: int = 10,
    keyword_fallback: Callable[[list[dict]], list[dict]] | None = None,
) -> list[dict]:
    """Use Claude Haiku to score articles 0-2 and return top max_items by score.

    min_score=2 keeps only direct holding news (score 1 = sector/macro context,
    which belongs in The World section, not Portfolio News).
    """
    if not articles:
        return articles

    def _fallback(exc: Exception) -> list[dict]:
        log.warning("Relevance filter failed (%s) — using %s",
                    exc, "keyword fallback" if keyword_fallback else "unfiltered list")
        base = keyword_fallback(articles) if keyword_fallback else articles
        return base[:max_items]

    try:
        ticker_name_list = ", ".join(
            f"{p['ticker']} ({p['name']})" for p in positions
        )
        articles_text = "\n".join(
            f"{i}: {a['headline']} [{a['source']}]"
            for i, a in enumerate(articles)
        )

        system_prompt = (
            "You are a financial news relevance classifier. Your only job is to decide "
            "whether a news article is relevant to an investor's stock portfolio. "
            "Return only valid JSON, no preamble, no markdown."
        )
        user_prompt = (
            f"A portfolio investor holds these tickers and company names:\n{ticker_name_list}\n\n"
            "Rate each of the following news articles for relevance to this portfolio on a scale of 0-2:\n"
            "- 0 = not relevant (gaming hardware reviews, entertainment, unrelated consumer products)\n"
            "- 1 = tangentially relevant (sector news, macro context)\n"
            "- 2 = directly relevant (specific company news, earnings, analyst ratings, "
            "product launches affecting holdings)\n\n"
            'Return a JSON array in this exact format:\n'
            '[{"index": 0, "score": 2}, ...]\n\n'
            f"Articles:\n{articles_text}"
        )

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        if response.stop_reason == "max_tokens":
            log.warning("Relevance filter response truncated (stop_reason=max_tokens) — falling back")
            return _fallback(RuntimeError("max_tokens"))

        raw = _extract_text_from_response(response).strip()
        # Strip markdown fences if model wraps output in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

        score_by_index: dict[int, int] = {}
        for entry in parsed:
            try:
                idx = int(entry["index"])
                score = int(entry["score"])
                if 0 <= idx < len(articles):
                    score_by_index[idx] = score
            except (KeyError, TypeError, ValueError):
                continue

        kept = [
            (i, a) for i, a in enumerate(articles)
            if score_by_index.get(i, 0) >= min_score
        ]
        kept.sort(key=lambda ia: score_by_index.get(ia[0], 0), reverse=True)
        filtered = [a for _, a in kept]

        log.info(
            "Relevance filter: %d -> %d articles (dropped %d below score %d)",
            len(articles), len(filtered), len(articles) - len(filtered), min_score,
        )
        return filtered[:max_items]

    except Exception as exc:
        return _fallback(exc)


_MACRO_KEYWORDS = [
    "fed", "inflation", "rate", "rates", "gdp", "election", "policy",
    "regulation", "tariff", "trade", "oil", "war", "treaty", "ai",
    "breakthrough", "discovery", "vaccine", "climate",
]
_EXCLUDE_KEYWORDS = [
    "airline", "airlines", "podcast", "podcasts", "celebrity", "sport", "sports",
]


def _compile_kw(keywords: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b")


_MACRO_RE = _compile_kw(_MACRO_KEYWORDS)
_EXCLUDE_RE = _compile_kw(_EXCLUDE_KEYWORDS)

_POLITICS_RE = _compile_kw([
    "election", "president", "senator", "government", "policy", "regulation",
    "tariff", "war", "treaty", "military", "sanction", "congress", "vote",
    "parliament", "ukraine", "gaza", "nato", "putin", "biden", "trump",
])
_SCIENCE_RE = _compile_kw([
    "ai", "breakthrough", "discovery", "vaccine", "climate", "research",
    "scientist", "satellite", "quantum", "study", "nasa", "spacex",
])
_ECONOMY_RE = _compile_kw([
    "fed", "inflation", "rate", "gdp", "trade", "oil", "market", "economy",
    "bank", "dollar", "euro", "recession", "growth", "stock",
])


def _matches_macro(title: str) -> bool:
    t = title.lower()
    if _EXCLUDE_RE.search(t):
        return False
    return bool(_MACRO_RE.search(t))


def _classify_news(title: str) -> str:
    t = title.lower()
    p = len(_POLITICS_RE.findall(t))
    s = len(_SCIENCE_RE.findall(t))
    e = len(_ECONOMY_RE.findall(t))
    if p >= s and p >= e and p > 0:
        return "POLITICS"
    if s > e and s > 0:
        return "SCIENCE"
    return "ECONOMY"


def _is_dup(title: str, kept: list[str], threshold: float = 0.8) -> bool:
    """Fuzzy duplicate check using token-sorted character ratio (stdlib only)."""

    def _norm(s: str) -> str:
        return " ".join(sorted(s.lower().split()))

    nt = _norm(title)
    return any(SequenceMatcher(None, nt, _norm(k)).ratio() >= threshold for k in kept)


_CLIMATE_QUERY = (
    "climate OR emissions OR renewable OR \"energy transition\" OR "
    "\"oil price\" OR \"natural gas\" OR \"power grid\" OR \"net zero\""
)

_TOP_HEADLINE_CATEGORIES = [
    ("business", "Economy/Markets"),
    ("science",  "Science/Tech"),
    ("health",   "Health"),
    ("general",  "Politics"),
]


def _select_world_news(articles: list[dict], max_items: int = 7) -> list[dict]:
    """Haiku pass: pick the most substantive, category-diverse world news items."""
    if not articles:
        return []

    articles_text = "\n".join(
        f"{i}: [{a.get('topic', a.get('category', ''))}] {a['headline']} [{a['source']}]"
        for i, a in enumerate(articles)
    )
    system_prompt = (
        "You are a world news editor curating a weekly digest. "
        "Return only valid JSON, no preamble, no markdown."
    )
    user_prompt = (
        f"Select the {max_items} most substantive, newsworthy articles from the list below.\n\n"
        "Keep: geopolitical events, economic data releases, scientific discoveries, "
        "policy decisions, market-moving news.\n"
        "Discard: lifestyle tips, celebrity/royal gossip, sports scores, product reviews, "
        "PR announcements, lottery results, recipes, wellness advice.\n"
        "Ensure variety — do not pick more than 2 articles from the same topic bucket.\n\n"
        f'Return a JSON array of selected indices only: [{{"index": 0}}, ...]\n\n'
        f"Articles:\n{articles_text}"
    )

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if response.stop_reason == "max_tokens":
            log.warning("World news selection truncated — using first %d articles", max_items)
            return articles[:max_items]

        raw = _extract_text_from_response(response).strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list, got {type(parsed).__name__}")

        selected: list[dict] = []
        for entry in parsed:
            try:
                idx = int(entry["index"])
                if 0 <= idx < len(articles):
                    selected.append(articles[idx])
            except (KeyError, TypeError, ValueError):
                continue

        return selected[:max_items]

    except Exception as exc:
        log.warning("World news selection failed (%s) — using first %d articles", exc, max_items)
        return articles[:max_items]


def fetch_general_news() -> list[dict]:
    """Curated world digest: per-category top-headlines + Haiku selection pass."""
    api_key = os.getenv("NEWS_API_KEY", "")
    newsapi = NewsApiClient(api_key=api_key)

    seen: list[str] = []
    articles: list[dict] = []

    def _collect(resp_articles, topic: str) -> list[dict]:
        out: list[dict] = []
        for a in resp_articles:
            title = (a.get("title") or "").strip()
            if not title or title == "[Removed]":
                continue
            if _is_dup(title, seen):
                continue
            seen.append(title)
            out.append(dict(
                headline=title,
                description=a.get("description") or "",
                source=a.get("source", {}).get("name", "Unknown"),
                date=(a.get("publishedAt") or "")[:10],
                url=a.get("url", ""),
                category=_classify_news(title),
                topic=topic,
            ))
        return out

    # Per-category top-headlines
    for api_cat, topic in _TOP_HEADLINE_CATEGORIES:
        try:
            resp = newsapi.get_top_headlines(category=api_cat, language="en", page_size=20)
            batch = _collect(resp.get("articles", []), topic)
            articles.extend(batch)
            log.debug("top-headlines category=%s: %d articles", api_cat, len(batch))
        except Exception as exc:
            log.warning("top-headlines category=%s failed: %s", api_cat, exc)

    # Climate/Energy via /everything (no native top-headlines category)
    try:
        resp = newsapi.get_everything(
            q=_CLIMATE_QUERY, language="en", sort_by="relevancy", page_size=20,
        )
        batch = _collect(resp.get("articles", []), "Climate/Energy")
        articles.extend(batch)
        log.debug("Climate/Energy query: %d articles", len(batch))
    except Exception as exc:
        log.warning("Climate/Energy query failed: %s", exc)

    log.info("fetch_general_news: %d deduplicated articles before selection", len(articles))

    selected = _select_world_news(articles, max_items=7)
    log.info("Fetched %d general news articles", len(selected))
    return selected
