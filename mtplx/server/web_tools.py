"""Server-hosted web tools for headless MTPLX.

Port of the native app's ``WebSearchService`` / ``URLFetcher`` /
``MTPLXChatToolFactory`` so ``mtplx serve --web`` (and quickstart) can
search and read pages without a Swift client in the loop.

Search is DuckDuckGo HTML + Brave HTML, URL-deduped, interleaved by
provider rank. No API keys. Fetch is a plain HTTP GET with readable-text
extraction and a 4000-character cap.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlparse

LOGGER = logging.getLogger("mtplx.web_tools")

HOSTED_WEB_TOOL_NAMES = frozenset({"web_search", "fetch_url"})
MAX_HOSTED_WEB_ROUNDS = 1
DEFAULT_SEARCH_RESULTS = 5
DEFAULT_FETCH_CHARACTERS = 4000
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
)
_HTTP_TIMEOUT_S = 12.0
_JACCARD_THRESHOLD = 0.85
_MAX_DUPLICATE_WARNINGS = 2

_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
_WS_RE = re.compile(r"\s+")
_DDG_ANCHOR_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_DDG_SNIPPET_RE = re.compile(
    r'<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
    re.DOTALL,
)
_BRAVE_BLOCK_RE = re.compile(
    r'<div[^>]*class="[^"]*snippet\s[^"]*"[^>]*data-pos="(\d+)"[^>]*>(.*?)</div>\s*</div>\s*</div>',
    re.DOTALL,
)
_BRAVE_TITLE_RE = re.compile(
    r'<div[^>]*class="[^"]*search-snippet-title[^"]*"[^>]*>(.*?)</div>',
    re.DOTALL,
)
_BRAVE_HREF_RE = re.compile(
    r'<a[^>]*href="(https?://[^"]+)"[^>]*class="[^"]*svelte-[^"]*"[^>]*>',
    re.DOTALL,
)
_BRAVE_CONTENT_RE = re.compile(
    r'<div[^>]*class="content[^"]*"[^>]*>(.*?)</div>',
    re.DOTALL,
)
_BRAVE_FALLBACK_TITLE_RE = re.compile(
    r'<div[^>]*class="[^"]*search-snippet-title[^"]*"[^>]*title="([^"]*)"[^>]*>',
    re.DOTALL,
)
_BRAVE_FALLBACK_HREF_RE = re.compile(
    r'<a[^>]*href="(https?://[^"]+)"[^>]*>[^<]*<div[^>]*class="[^"]*site-name',
    re.DOTALL,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)
_NOSCRIPT_RE = re.compile(r"<noscript\b[^>]*>.*?</noscript>", re.IGNORECASE | re.DOTALL)


def add_web_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--web",
        action="store_true",
        default=False,
        help=(
            "Host web_search and fetch_url on the server (DuckDuckGo + Brave). "
            "Injected when the client sends no tools; the server executes the "
            "loop so curl / Studio get live sources."
        ),
    )


def hosted_web_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web and automatically read the strongest current "
                    "sources, including the homepage of any domain the user named. "
                    "Identify a named site from its own homepage before listing "
                    "alternatives. Do not substitute a similarly named product "
                    "(example: kink.com the studio is not getkink.com the dating app). "
                    "Cite source URLs from the tool results. If the homepage was "
                    "not fetched, say so instead of guessing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": (
                    "Fetch and extract readable text content from a URL. Use this "
                    "when the user provides a URL and wants to know what is on "
                    "that page."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch",
                        }
                    },
                    "required": ["url"],
                },
            },
        },
    ]


def _tool_name(tool: dict[str, Any] | None) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
    else:
        name = tool.get("name")
    return str(name or "").strip()


def should_host_web_tools(tools: list[dict[str, Any]] | None) -> bool:
    names = {_tool_name(tool) for tool in (tools or [])}
    names.discard("")
    return bool(names) and names <= HOSTED_WEB_TOOL_NAMES


def hosted_tool_call_name(tool_call: dict[str, Any] | None) -> str:
    if not isinstance(tool_call, dict):
        return ""
    function = tool_call.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    return str(tool_call.get("name") or "").strip()


def hosted_tool_call_arguments(tool_call: dict[str, Any] | None) -> str:
    if not isinstance(tool_call, dict):
        return ""
    function = tool_call.get("function")
    if isinstance(function, dict):
        arguments = function.get("arguments")
    else:
        arguments = tool_call.get("arguments")
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    return str(arguments or "")


def all_tool_calls_are_hosted(tool_calls: Iterable[dict[str, Any]] | None) -> bool:
    calls = list(tool_calls or [])
    if not calls:
        return False
    return all(hosted_tool_call_name(call) in HOSTED_WEB_TOOL_NAMES for call in calls)


_DOMAIN_RE = re.compile(
    r"\b(?:https?://)?((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,})\b",
    re.IGNORECASE,
)
_SKIP_DOMAIN_SUFFIXES = frozenset(
    {
        "example.com",
        "localhost",
        "local",
        "test",
        "invalid",
    }
)

HOSTED_WEB_CONTRACT = (
    "Live web tools are on. If the user names a site or domain, identify it "
    "from that site's own homepage in the tool results before listing "
    "alternatives. Do not swap in a similarly named product. Cite URLs from "
    "the tool results. If a named homepage is missing, say so."
)


def named_query_domains(query: str) -> list[str]:
    """Bare domains the user typed (kink.com), not search-operator noise."""

    seen: set[str] = set()
    ordered: list[str] = []
    for match in _DOMAIN_RE.finditer(query or ""):
        host = match.group(1).lower().removeprefix("www.")
        if host in seen or host in _SKIP_DOMAIN_SUFFIXES:
            continue
        if host.endswith(".png") or host.endswith(".jpg") or host.endswith(".gif"):
            continue
        seen.add(host)
        ordered.append(host)
    return ordered


def named_query_homepage_urls(query: str) -> list[str]:
    return [f"https://{host}/" for host in named_query_domains(query)]


def named_query_grounding_urls(query: str) -> list[str]:
    """Homepage, www twin, and the Wikipedia article titled like the domain."""

    urls: list[str] = []
    seen: set[str] = set()
    for host in named_query_domains(query):
        wiki_title = urllib.parse.quote(host[:1].upper() + host[1:])
        for url in (
            f"https://{host}/",
            f"https://www.{host}/",
            f"https://en.wikipedia.org/wiki/{wiki_title}",
        ):
            key = url.rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            urls.append(url)
    return urls


_ACCESS_GATE_MARKERS = (
    "privacy policy acceptance",
    "adults only",
    "you certify that you are 18",
    "age verification",
    "you must be 18",
)


def _looks_like_access_gate(title: str, content: str) -> bool:
    blob = f"{title} {content}".lower()
    return any(marker in blob for marker in _ACCESS_GATE_MARKERS)


def fallback_web_search_tool_calls(query: str) -> list[dict[str, Any]]:
    """Synthesize a web_search call when the model opened tools but sent no name."""

    cleaned = str(query or "").strip()
    if not cleaned:
        return []
    return [
        {
            "id": "call_web_search_fallback",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": json.dumps({"query": cleaned}, ensure_ascii=False),
            },
        }
    ]


def usable_hosted_tool_calls(
    tool_calls: Iterable[dict[str, Any]] | None,
    *,
    fallback_query: str,
    saw_tool_attempt: bool = False,
    finish_reason: str | None = None,
) -> list[dict[str, Any]]:
    calls = [call for call in (tool_calls or []) if hosted_tool_call_name(call)]
    if all_tool_calls_are_hosted(calls):
        return calls
    if saw_tool_attempt or str(finish_reason or "") == "tool_calls":
        return fallback_web_search_tool_calls(fallback_query)
    return []


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class URLFetchResult:
    url: str
    title: str | None
    content: str


@dataclass
class _MergedCandidate:
    provider: str
    provider_rank: int
    result: WebSearchResult

    @property
    def sort_priority(self) -> int:
        return 0 if self.provider == "duckduckgo" else 1


class ToolSessionState:
    """Jaccard duplicate-query guard. Mirrors the Swift ``ToolSessionState``."""

    def __init__(self) -> None:
        self.seen_queries: list[str] = []
        self.warning_count = 0
        self.disabled = False

    def reset(self) -> None:
        self.seen_queries.clear()
        self.warning_count = 0
        self.disabled = False

    def begin(self, query: str) -> tuple[str, str | None, int]:
        """Return ``("proceed", None, 0)``, ``("duplicate", previous, n)``, or ``("disabled", "", n)``."""
        if self.disabled:
            return ("disabled", "", self.warning_count)
        normalized = query.lower().strip()
        if not normalized:
            return ("proceed", None, 0)
        for previous in self.seen_queries:
            if _jaccard(previous, normalized) >= _JACCARD_THRESHOLD:
                self.warning_count += 1
                if self.warning_count >= _MAX_DUPLICATE_WARNINGS:
                    self.disabled = True
                return ("duplicate", previous, self.warning_count)
        self.seen_queries.append(normalized)
        return ("proceed", None, 0)


def _jaccard(lhs: str, rhs: str) -> float:
    left = set(lhs.split())
    right = set(rhs.split())
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union


def decode_html_entities(value: str) -> str:
    return (
        html.unescape(value)
        .replace("&nbsp;", " ")
    )


def normalize_whitespace(value: str) -> str:
    return _WS_RE.sub(" ", value).strip()


def clean_html_fragment(fragment: str) -> str:
    return normalize_whitespace(decode_html_entities(_TAG_RE.sub(" ", fragment)))


def is_ddg_bot_detection(html_text: str) -> bool:
    return (
        "cc=botnet" in html_text
        or "anomaly-modal" in html_text
        or "bots use DuckDuckGo" in html_text
    )


def resolve_ddg_result_url(href: str) -> str | None:
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    uddg = query.get("uddg", [None])[0]
    if uddg:
        return urllib.parse.unquote(uddg)
    cleaned = decode_html_entities(href).strip()
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return cleaned
    return None


def parse_ddg_results(html_text: str, limit: int) -> list[WebSearchResult]:
    if limit <= 0:
        return []
    anchors = list(_DDG_ANCHOR_RE.finditer(html_text))
    if not anchors:
        return []
    results: list[WebSearchResult] = []
    for match in anchors[:limit]:
        resolved = resolve_ddg_result_url(match.group(1))
        if not resolved:
            continue
        title = clean_html_fragment(match.group(2))
        rest = html_text[match.end() :]
        snippet_match = _DDG_SNIPPET_RE.search(rest)
        snippet = clean_html_fragment(snippet_match.group(1)) if snippet_match else ""
        results.append(WebSearchResult(title=title, url=resolved, snippet=snippet))
    return results


def parse_brave_fallback(html_text: str, limit: int) -> list[WebSearchResult]:
    titles = list(_BRAVE_FALLBACK_TITLE_RE.finditer(html_text))
    hrefs = list(_BRAVE_FALLBACK_HREF_RE.finditer(html_text))
    count = min(len(titles), len(hrefs), max(limit, 0))
    results: list[WebSearchResult] = []
    for index in range(count):
        url = titles and hrefs[index].group(1)
        if not url:
            continue
        title = decode_html_entities(titles[index].group(1))
        results.append(WebSearchResult(title=title, url=url, snippet=""))
    return results


def parse_brave_results(html_text: str, limit: int) -> list[WebSearchResult]:
    if limit <= 0:
        return []
    blocks = list(_BRAVE_BLOCK_RE.finditer(html_text))
    if not blocks:
        return parse_brave_fallback(html_text, limit)
    results: list[WebSearchResult] = []
    for block in blocks[:limit]:
        body = block.group(2)
        title_match = _BRAVE_TITLE_RE.search(body)
        href_match = _BRAVE_HREF_RE.search(body)
        if not title_match or not href_match:
            continue
        content_match = _BRAVE_CONTENT_RE.search(body)
        results.append(
            WebSearchResult(
                title=clean_html_fragment(title_match.group(1)),
                url=href_match.group(1),
                snippet=clean_html_fragment(content_match.group(1)) if content_match else "",
            )
        )
    return results


def preferred_candidate(
    current: _MergedCandidate, replacement: _MergedCandidate
) -> _MergedCandidate:
    if current.provider_rank != replacement.provider_rank:
        return current if current.provider_rank < replacement.provider_rank else replacement
    if len(current.result.snippet) != len(replacement.result.snippet):
        return (
            current
            if len(current.result.snippet) > len(replacement.result.snippet)
            else replacement
        )
    return current if current.sort_priority <= replacement.sort_priority else replacement


def merge_provider_batches(
    batches: list[tuple[str, list[WebSearchResult]]],
    max_results: int,
) -> list[WebSearchResult]:
    if max_results <= 0:
        return []
    deduped: dict[str, _MergedCandidate] = {}
    for provider, results in batches:
        for rank, result in enumerate(results):
            candidate = _MergedCandidate(provider=provider, provider_rank=rank, result=result)
            existing = deduped.get(result.url)
            if existing is None:
                deduped[result.url] = candidate
            else:
                deduped[result.url] = preferred_candidate(existing, candidate)
    grouped: dict[int, list[_MergedCandidate]] = {}
    for candidate in deduped.values():
        grouped.setdefault(candidate.provider_rank, []).append(candidate)
    merged: list[WebSearchResult] = []
    for rank in sorted(grouped):
        group = sorted(
            grouped[rank],
            key=lambda item: (
                -len(item.result.snippet),
                item.sort_priority,
                item.result.url,
            ),
        )
        for candidate in group:
            merged.append(candidate.result)
            if len(merged) == max_results:
                return merged
    return merged


def extract_html_title(html_text: str) -> str | None:
    match = _TITLE_RE.search(html_text)
    if not match:
        return None
    title = clean_html_fragment(match.group(1))
    return title or None


def extract_readable_content(html_text: str) -> str:
    stripped = _NOSCRIPT_RE.sub(" ", _STYLE_RE.sub(" ", _SCRIPT_RE.sub(" ", html_text)))
    for pattern in (r"</p>", r"<br\s*/?>", r"</div>", r"</li>", r"</h[1-6]>"):
        stripped = re.sub(pattern, "\n", stripped, flags=re.IGNORECASE)
    stripped = decode_html_entities(_TAG_RE.sub(" ", stripped))
    lines = [normalize_whitespace(line) for line in stripped.splitlines()]
    return "\n".join(line for line in lines if line)


def cap_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


class WebTransport:
    def fetch(self, url: str, *, headers: dict[str, str]) -> tuple[int, str, str]:
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
                status = int(getattr(response, "status", 200) or 200)
                content_type = str(response.headers.get("Content-Type") or "")
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise WebToolsError(f"HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise WebToolsError(str(exc.reason or exc)) from exc
        return status, content_type, body.decode("utf-8", errors="replace")


class WebToolsError(RuntimeError):
    pass


def _browser_headers(*, referer: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def _http_get(transport: WebTransport, url: str, *, referer: str | None = None) -> str:
    status, _content_type, body = transport.fetch(url, headers=_browser_headers(referer=referer))
    if not 200 <= status <= 299:
        raise WebToolsError(f"HTTP {status}")
    return body


def search_duckduckgo(
    query: str,
    *,
    max_results: int,
    transport: WebTransport,
) -> list[WebSearchResult]:
    endpoints = (
        "https://html.duckduckgo.com/html/",
        "https://lite.duckduckgo.com/lite/",
    )
    encoded = urllib.parse.urlencode({"q": query})
    for endpoint in endpoints:
        url = f"{endpoint}?{encoded}"
        try:
            body = _http_get(transport, url, referer="https://duckduckgo.com/")
        except WebToolsError as exc:
            LOGGER.info("DuckDuckGo %s failed: %s", endpoint, exc)
            continue
        if is_ddg_bot_detection(body):
            LOGGER.info("DuckDuckGo bot detection on %s", endpoint)
            continue
        results = parse_ddg_results(body, max_results)
        if results:
            return results
    return []


def search_brave(
    query: str,
    *,
    max_results: int,
    transport: WebTransport,
) -> list[WebSearchResult]:
    encoded = urllib.parse.urlencode({"q": query, "source": "web"})
    url = f"https://search.brave.com/search?{encoded}"
    body = _http_get(transport, url)
    return parse_brave_results(body, max_results)


_SEARCH_CACHE: dict[str, list[WebSearchResult]] = {}
_FETCH_CACHE: dict[str, URLFetchResult] = {}


def web_search(
    query: str,
    *,
    max_results: int = DEFAULT_SEARCH_RESULTS,
    transport: WebTransport | None = None,
) -> list[WebSearchResult]:
    cache_key = query.lower().strip()
    cached = _SEARCH_CACHE.get(cache_key)
    if cached is not None:
        return cached[: max(max_results, 0)]
    limit = max(max_results, 0)
    if limit == 0:
        return []
    provider_limit = max(limit, 10)
    transport = transport or WebTransport()

    def _ddg() -> list[WebSearchResult]:
        try:
            return search_duckduckgo(query, max_results=provider_limit, transport=transport)
        except Exception as exc:  # noqa: BLE001 — provider isolation
            LOGGER.info("DuckDuckGo search failed: %s", exc)
            return []

    def _brave() -> list[WebSearchResult]:
        try:
            return search_brave(query, max_results=provider_limit, transport=transport)
        except Exception as exc:  # noqa: BLE001 — provider isolation
            LOGGER.info("Brave search failed: %s", exc)
            return []

    with ThreadPoolExecutor(max_workers=2) as pool:
        ddg_future = pool.submit(_ddg)
        brave_future = pool.submit(_brave)
        batches = [
            ("duckduckgo", ddg_future.result()),
            ("brave", brave_future.result()),
        ]
    merged = merge_provider_batches(batches, provider_limit)
    _SEARCH_CACHE[cache_key] = merged
    return merged[:limit]


def fetch_url(
    url: str,
    *,
    max_characters: int = DEFAULT_FETCH_CHARACTERS,
    transport: WebTransport | None = None,
) -> URLFetchResult:
    cache_key = f"{max_characters}|{url}"
    cached = _FETCH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    transport = transport or WebTransport()
    status, content_type, body = transport.fetch(url, headers=_browser_headers())
    if not 200 <= status <= 299:
        raise WebToolsError(f"HTTP {status}")
    lowered = content_type.lower()
    if "html" in lowered or "<html" in body or "<body" in body:
        result = URLFetchResult(
            url=url,
            title=extract_html_title(body),
            content=cap_text(extract_readable_content(body), max_characters),
        )
    else:
        host = urlparse(url).hostname
        result = URLFetchResult(
            url=url,
            title=host,
            content=cap_text(normalize_whitespace(body), max_characters),
        )
    _FETCH_CACHE[cache_key] = result
    return result


def _json_object(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _parse_string_arg(arguments_json: str, key: str) -> str:
    try:
        parsed = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    value = parsed.get(key)
    return str(value or "").strip()


def dispatch_hosted_web_tool(
    name: str,
    arguments_json: str,
    *,
    session: ToolSessionState | None = None,
    transport: WebTransport | None = None,
    search: Callable[..., list[WebSearchResult]] | None = None,
    fetch: Callable[..., URLFetchResult] | None = None,
) -> str:
    if name == "web_search":
        return _dispatch_web_search(
            arguments_json,
            session=session or ToolSessionState(),
            transport=transport,
            search=search or web_search,
            fetch=fetch or fetch_url,
        )
    if name == "fetch_url":
        return _dispatch_fetch_url(
            arguments_json,
            transport=transport,
            fetch=fetch or fetch_url,
        )
    return _json_object(
        {
            "error": "unknown_tool",
            "tool": name,
            "note": "Tool not implemented in MTPLX chat; answer from knowledge.",
        }
    )


def _dispatch_web_search(
    arguments_json: str,
    *,
    session: ToolSessionState,
    transport: WebTransport | None,
    search: Callable[..., list[WebSearchResult]],
    fetch: Callable[..., URLFetchResult],
) -> str:
    query = _parse_string_arg(arguments_json, "query")
    if not query:
        return _json_object(
            {
                "error": "empty_query",
                "note": "web_search called with empty query; answer from knowledge.",
            }
        )
    decision, previous, warning_count = session.begin(query)
    if decision == "disabled":
        return _json_object(
            {
                "query": query,
                "note": (
                    "web_search is disabled for the rest of this turn. "
                    "Answer from knowledge or the previously fetched sources."
                ),
            }
        )
    if decision == "duplicate":
        return _json_object(
            {
                "query": query,
                "previous_query": previous,
                "warning_count": warning_count,
                "note": (
                    "Query is too similar to a previous search this turn. "
                    "Use the earlier results instead of repeating the search."
                ),
            }
        )
    try:
        results = search(query, max_results=DEFAULT_SEARCH_RESULTS, transport=transport)
    except Exception as exc:  # noqa: BLE001 — tool result, not a crash
        LOGGER.info("web_search failed: %s", exc)
        return _json_object(
            {
                "query": query,
                "error": "search_failed",
                "detail": str(exc),
                "note": "Search backend errored; answer from knowledge and do not retry.",
            }
        )

    identity: list[WebSearchResult] = []
    for host in named_query_domains(query):
        extras = [host] if query.strip().lower() != host else []
        extras.append(f"{host} wikipedia")
        for extra in extras:
            try:
                identity.extend(search(extra, max_results=3, transport=transport))
            except Exception as exc:  # noqa: BLE001 — keep the user's query results
                LOGGER.info("identity search failed for %s: %s", extra, exc)
    if identity:
        seen_urls = {item.url.rstrip("/") for item in identity}
        results = identity + [item for item in results if item.url.rstrip("/") not in seen_urls]

    named_urls = named_query_grounding_urls(query)
    fetched: dict[str, URLFetchResult] = {}
    fetch_targets = list(named_urls) + [item.url for item in results[: min(3, len(results))]]

    def _one(url: str) -> tuple[str, URLFetchResult | None]:
        try:
            return url, fetch(url, transport=transport)
        except Exception:  # noqa: BLE001 — skip unread pages
            return url, None

    if fetch_targets:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for url, page in pool.map(_one, fetch_targets):
                if page is not None:
                    fetched[url] = page

    if not results and not fetched:
        return _json_object(
            {
                "query": query,
                "results": [],
                "note": "No results. Answer the user's question from your knowledge.",
            }
        )

    payload_results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for url in named_urls:
        page = fetched.get(url)
        host = urlparse(url).hostname or ""
        item: dict[str, Any] = {
            "title": (page.title if page is not None else host) or host,
            "url": url,
            "snippet": "",
            "host": host,
            "named_site": True,
        }
        if page is not None:
            item["page_content"] = page.content
            if _looks_like_access_gate(page.title, page.content):
                item["access_gate"] = True
        else:
            item["error"] = "named_homepage_unreadable"
        if "wikipedia.org" in host or "wikipedia.org" in url:
            item["identity_source"] = "wikipedia"
        payload_results.append(item)
        seen_urls.add(url.rstrip("/"))

    for result in results:
        if result.url.rstrip("/") in seen_urls:
            continue
        item = {
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "host": urlparse(result.url).hostname or "",
        }
        page = fetched.get(result.url)
        if page is not None:
            item["page_content"] = page.content
        payload_results.append(item)

    return _json_object(
        {
            "query": query,
            "results": payload_results,
            "note": (
                "Results marked named_site are the user's own domains plus Wikipedia "
                "identity pages, fetched first. Identify those sites from page_content "
                "before listing alternatives. Ignore access_gate pages as product copy. "
                "Do not substitute a similarly named product. Cite URLs from this payload."
            ),
        }
    )


def _dispatch_fetch_url(
    arguments_json: str,
    *,
    transport: WebTransport | None,
    fetch: Callable[..., URLFetchResult],
) -> str:
    raw_url = _parse_string_arg(arguments_json, "url")
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return _json_object(
            {
                "error": "invalid_url",
                "url": raw_url,
                "note": "fetch_url requires an http(s) URL.",
            }
        )
    try:
        page = fetch(raw_url, transport=transport)
    except Exception as exc:  # noqa: BLE001 — tool result, not a crash
        LOGGER.info("fetch_url failed for %s: %s", raw_url, exc)
        return _json_object(
            {
                "error": "fetch_failed",
                "url": raw_url,
                "detail": str(exc),
                "note": "Could not fetch the URL; do not retry the same URL this turn.",
            }
        )
    return _json_object(
        {
            "url": page.url,
            "title": page.title or "",
            "content": page.content,
        }
    )


@dataclass
class AccumulatedToolCall:
    index: int
    call_id: str | None = None
    name: str | None = None
    arguments: str = ""


def accumulate_tool_call_delta(
    acc: list[AccumulatedToolCall],
    *,
    index: int,
    call_id: str | None,
    name: str | None,
    arguments_delta: str,
) -> None:
    while len(acc) <= index:
        acc.append(AccumulatedToolCall(index=len(acc)))
    item = acc[index]
    if call_id:
        item.call_id = call_id
    if name:
        item.name = name
    if arguments_delta:
        item.arguments += arguments_delta


def tool_calls_from_accumulator(
    acc: list[AccumulatedToolCall],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in acc:
        name = (item.name or "").strip()
        if not name:
            continue
        call_id = item.call_id or f"call_{item.index}"
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": item.arguments or "{}"},
            }
        )
    return calls
