from __future__ import annotations

import json

from mtplx.server.web_tools import (
    ToolSessionState,
    WebSearchResult,
    all_tool_calls_are_hosted,
    clean_html_fragment,
    dispatch_hosted_web_tool,
    extract_readable_content,
    hosted_web_tool_definitions,
    merge_provider_batches,
    parse_brave_results,
    parse_ddg_results,
    resolve_ddg_result_url,
    should_host_web_tools,
    tool_calls_from_accumulator,
    accumulate_tool_call_delta,
)


DDG_HTML = """
<a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fkink.com%2F">Kink.com</a>
<a class="result__snippet">Official site</a>
<a class="result__a" href="https://example.com/alt">Alt studio</a>
<div class="result__snippet">Another studio</div>
"""

BRAVE_HTML = """
<div class="snippet foo" data-pos="1"><div class="search-snippet-title">Kink.com</div>
<a href="https://kink.com/" class="svelte-abc"></a>
<div class="content">Official site</div>
</div></div></div>
<div class="snippet foo" data-pos="2"><div class="search-snippet-title">Alt studio</div>
<a href="https://example.com/alt" class="svelte-abc"></a>
<div class="content">Another studio</div>
</div></div></div>
"""


def test_parse_ddg_results_resolves_uddg_and_direct_urls():
    results = parse_ddg_results(DDG_HTML, 5)
    assert [item.url for item in results] == [
        "https://kink.com/",
        "https://example.com/alt",
    ]
    assert results[0].title == "Kink.com"
    assert results[0].snippet == "Official site"
    assert results[1].snippet == "Another studio"


def test_parse_brave_results_reads_snippet_blocks():
    results = parse_brave_results(BRAVE_HTML, 5)
    assert [item.title for item in results] == ["Kink.com", "Alt studio"]
    assert results[0].url == "https://kink.com/"


def test_merge_provider_batches_dedupes_by_url():
    merged = merge_provider_batches(
        [
            (
                "duckduckgo",
                [WebSearchResult("Kink", "https://kink.com/", "short")],
            ),
            (
                "brave",
                [WebSearchResult("Kink", "https://kink.com/", "much longer snippet")],
            ),
        ],
        5,
    )
    assert len(merged) == 1
    assert merged[0].snippet == "much longer snippet"


def test_resolve_ddg_result_url():
    assert (
        resolve_ddg_result_url("https://duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example")
        == "https://a.example"
    )


def test_clean_and_extract_html():
    assert clean_html_fragment("<b>Hello&nbsp;world</b>") == "Hello world"
    text = extract_readable_content(
        "<html><script>hide()</script><p>Hello</p><div>World</div></html>"
    )
    assert "Hello" in text
    assert "World" in text
    assert "hide()" not in text


def test_named_query_domains_and_homepage_fetch_come_first():
    from mtplx.server.web_tools import (
        URLFetchResult,
        WebSearchResult,
        dispatch_hosted_web_tool,
        named_query_domains,
    )

    assert named_query_domains("alternatywy dla kink.com na 09/2026") == ["kink.com"]

    seen_queries: list[str] = []

    def fake_search(query, **_kwargs):
        seen_queries.append(query)
        return [WebSearchResult("Dating SEO", "https://listicle.example/kink", "apps")]

    def fake_fetch(url, **_kwargs):
        if "wikipedia.org" in url:
            return URLFetchResult(
                url=url,
                title="Kink.com - Wikipedia",
                content="Kink.com is a San Francisco BDSM film studio.",
            )
        if "kink.com" in url:
            return URLFetchResult(
                url=url,
                title="Kink - Privacy Policy Acceptance",
                content="Warning: Adults Only. You certify that you are 18 years old.",
            )
        return URLFetchResult(url=url, title="list", content="dating apps")

    payload = json.loads(
        dispatch_hosted_web_tool(
            "web_search",
            '{"query":"alternatywy dla kink.com"}',
            search=fake_search,
            fetch=fake_fetch,
        )
    )
    assert payload["results"][0]["named_site"] is True
    assert payload["results"][0]["url"] == "https://kink.com/"
    assert payload["results"][0].get("access_gate") is True
    wiki = next(item for item in payload["results"] if item.get("identity_source") == "wikipedia")
    assert "BDSM film studio" in wiki["page_content"]
    assert "named_site" in payload["note"]
    assert "kink.com" in seen_queries
    assert any("wikipedia" in query for query in seen_queries)


def test_usable_hosted_tool_calls_falls_back_when_names_are_empty():
    from mtplx.server.web_tools import usable_hosted_tool_calls

    empty = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        }
    ]
    fallback = usable_hosted_tool_calls(
        empty,
        fallback_query="alternatywy kink.com 09/2026",
        saw_tool_attempt=True,
        finish_reason="tool_calls",
    )
    assert fallback[0]["function"]["name"] == "web_search"
    assert "kink.com" in fallback[0]["function"]["arguments"]


def test_rebinding_cancel_binder_forwards_to_latest_generation():
    from mtplx.server.openai import _rebinding_cancel_binder

    calls: list[str] = []
    registered: list[object] = []

    def bind_once(cancel):
        if registered:
            raise RuntimeError("already bound")
        registered.append(cancel)
        return True

    binder = _rebinding_cancel_binder(bind_once)
    assert binder is not None
    binder(lambda: calls.append("first"))
    binder(lambda: calls.append("second"))
    registered[0]()
    assert calls == ["second"]


def test_hosted_web_stream_hides_empty_function_calls_and_searches(monkeypatch):
    import asyncio

    from mtplx.server.core import (
        OutputItemStarted,
        ReasoningDelta,
        TextDelta,
        ToolCallDelta,
        TurnCompleted,
        TurnStarted,
    )
    from mtplx.server import openai as openai_mod
    from mtplx.server.openai import (
        ChatCompletionRequest,
        ChatMessage,
        _hosted_web_turn_events,
    )

    monkeypatch.setattr(
        openai_mod,
        "dispatch_hosted_web_tool",
        lambda *_args, **_kwargs: '{"query":"kink.com","results":[{"url":"https://example.com"}]}',
    )
    rounds: list[int] = []

    class FakeStream:
        def __init__(self, events):
            self._events = events

        async def events(self):
            for event in self._events:
                yield event

    async def fake_impl(_raw, request, *, return_turn, bind_cancel=None):
        rounds.append(len(request.messages))
        if not any(getattr(message, "role", None) == "tool" for message in request.messages):
            return FakeStream(
                [
                    TurnStarted(response_id="c1", model="buddy", created=1),
                    ReasoningDelta(delta="I should search."),
                    OutputItemStarted(kind="tool", index=0),
                    OutputItemStarted(kind="tool", index=1),
                    ToolCallDelta(index=0, call_id="call_a", name="", arguments_delta=""),
                    ToolCallDelta(index=1, call_id="call_b", name="", arguments_delta=""),
                    TurnCompleted(finish_reason="tool_calls"),
                ]
            )
        return FakeStream(
            [
                TurnStarted(response_id="c2", model="buddy", created=2),
                TextDelta(delta="1. example.com\n"),
                TurnCompleted(finish_reason="stop"),
            ]
        )

    request = ChatCompletionRequest(
        messages=[
            ChatMessage(
                role="user",
                content="ello! potrzebuję alternatyw dla kink.com",
            )
        ],
        tools=[{"type": "function", "function": {"name": "web_search"}}],
        stream=True,
    )

    async def collect():
        events = []
        async for event in _hosted_web_turn_events(
            fake_impl, None, request, bind_cancel=None
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())
    assert not any(isinstance(event, OutputItemStarted) and event.kind == "tool" for event in events)
    assert not any(isinstance(event, ToolCallDelta) for event in events)
    assert any(isinstance(event, TextDelta) and "example.com" in event.delta for event in events)
    assert events[-1].finish_reason == "stop"
    assert rounds == [2, 4]


def test_hosted_web_stream_releases_direct_answer_without_tool_call():
    import asyncio

    from mtplx.server.core import (
        OutputItemStarted,
        ReasoningDelta,
        TextDelta,
        TurnCompleted,
        TurnStarted,
    )
    from mtplx.server.openai import (
        ChatCompletionRequest,
        ChatMessage,
        _hosted_web_turn_events,
    )

    rounds: list[int] = []

    class FakeStream:
        async def events(self):
            for event in [
                TurnStarted(response_id="c1", model="buddy", created=1),
                ReasoningDelta(delta="Simple greeting, no tool needed."),
                OutputItemStarted(kind="text", index=0),
                TextDelta(delta="Hi Maciej! "),
                TextDelta(delta="Enjoy the Chaconne."),
                TurnCompleted(finish_reason="stop"),
            ]:
                yield event

    async def fake_impl(_raw, request, *, return_turn, bind_cancel=None):
        rounds.append(len(request.messages))
        return FakeStream()

    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Say hi in one sentence.")],
        stream=True,
    )

    async def collect():
        return [
            event
            async for event in _hosted_web_turn_events(
                fake_impl, None, request, bind_cancel=None
            )
        ]

    events = asyncio.run(collect())
    text = "".join(event.delta for event in events if isinstance(event, TextDelta))
    assert text == "Hi Maciej! Enjoy the Chaconne."
    assert isinstance(events[-1], TurnCompleted)
    assert events[-1].finish_reason == "stop"
    assert rounds == [2]


def test_hosted_tool_definitions_and_hosting_gate():
    tools = hosted_web_tool_definitions()
    assert should_host_web_tools(tools)
    assert {item["function"]["name"] for item in tools} == {"web_search", "fetch_url"}
    assert not should_host_web_tools(
        tools + [{"type": "function", "function": {"name": "bash"}}]
    )
    assert all_tool_calls_are_hosted(
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{}"},
            }
        ]
    )


def test_dispatch_web_search_empty_and_success():
    empty = json.loads(dispatch_hosted_web_tool("web_search", "{}"))
    assert empty["error"] == "empty_query"

    def fake_search(query, **_kwargs):
        return [WebSearchResult("Kink", "https://kink.com/", "Official")]

    def fake_fetch(url, **_kwargs):
        from mtplx.server.web_tools import URLFetchResult

        return URLFetchResult(url=url, title="Kink", content="page body")

    payload = json.loads(
        dispatch_hosted_web_tool(
            "web_search",
            '{"query":"kink alternatives"}',
            search=fake_search,
            fetch=fake_fetch,
        )
    )
    assert payload["query"] == "kink alternatives"
    assert payload["results"][0]["url"] == "https://kink.com/"
    assert payload["results"][0]["page_content"] == "page body"


def test_duplicate_query_guard():
    session = ToolSessionState()

    def fake_fetch(url, **_kwargs):
        from mtplx.server.web_tools import URLFetchResult

        return URLFetchResult(url=url, title="A", content="body")

    first = json.loads(
        dispatch_hosted_web_tool(
            "web_search",
            '{"query":"same query here"}',
            session=session,
            search=lambda *_args, **_kwargs: [
                WebSearchResult("A", "https://a.example", "s")
            ],
            fetch=fake_fetch,
        )
    )
    assert "results" in first
    second = json.loads(
        dispatch_hosted_web_tool(
            "web_search",
            '{"query":"same query here"}',
            session=session,
            search=lambda *_args, **_kwargs: [],
        )
    )
    assert "previous_query" in second


def test_cli_and_server_accept_web_flag():
    from mtplx.cli import build_parser
    from mtplx.server.openai import parse_args

    quickstart = build_parser().parse_args(["quickstart", "--web"])
    serve = build_parser().parse_args(["serve", "--yes", "--web"])
    daemon = parse_args(["--model", "m", "--web"])
    assert quickstart.web is True
    assert serve.web is True
    assert daemon.web is True
    assert build_parser().parse_args(["quickstart"]).web is False


def test_hosted_web_json_loop_executes_then_answers(monkeypatch):
    import asyncio

    from fastapi.responses import JSONResponse

    from mtplx.server import openai as openai_mod
    from mtplx.server.openai import (
        ChatCompletionRequest,
        ChatMessage,
        _run_hosted_web_chat_completions,
    )

    monkeypatch.setattr(
        openai_mod,
        "dispatch_hosted_web_tool",
        lambda *_args, **_kwargs: '{"query":"x","results":[]}',
    )

    rounds: list[list[str]] = []

    async def fake_impl(_raw, request, *, return_turn, bind_cancel=None):
        roles = [message.role for message in request.messages]
        rounds.append(roles)
        if "tool" not in roles:
            return JSONResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": '{"query":"x"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        return JSONResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "done"},
                    }
                ]
            }
        )

    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        tools=[
            {
                "type": "function",
                "function": {"name": "web_search", "parameters": {"type": "object"}},
            }
        ],
        stream=False,
    )
    response = asyncio.run(
        _run_hosted_web_chat_completions(
            fake_impl,
            None,
            request,
            return_turn=False,
        )
    )
    body = json.loads(response.body)
    assert body["choices"][0]["message"]["content"] == "done"
    assert rounds[0] == ["system", "user"]
    assert rounds[1][-1] == "tool"


def test_accumulate_tool_call_deltas():
    acc = []
    accumulate_tool_call_delta(
        acc, index=0, call_id="call_1", name="web_search", arguments_delta='{"q'
    )
    accumulate_tool_call_delta(
        acc, index=0, call_id=None, name=None, arguments_delta='uery":"x"}'
    )
    calls = tool_calls_from_accumulator(acc)
    assert calls[0]["function"]["name"] == "web_search"
    assert calls[0]["function"]["arguments"] == '{"query":"x"}'
