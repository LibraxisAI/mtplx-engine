#!/usr/bin/env python3
"""Agent-prompt fidelity audit: is the prompt MTPLX feeds the model in a long
agent session, token id for token id, what the model's own chat template
produces?

No model weights are loaded. For every (pack, client lane, session, thinking
mode, turn boundary) two id sequences are built from the SAME OpenAI-style
request body:

  REFERENCE  the pack's own chat template rendered through transformers'
             ``apply_chat_template`` (tools passed as sent, tool-call
             arguments parsed from their JSON strings the way every reference
             serving stack does), with the thinking switches the server
             resolved, then encoded with the pack's ``tokenizer.json`` under
             the pre-tokenizer regex the pack declares
             (``tokenizer_config.json: pretokenize_regex``).

  MTPLX      the server's request path, called in the order
             ``/v1/chat/completions`` calls it (mtplx/server/openai.py,
             ``chat_completions``): ``resolve_request_policy`` ->
             ``_vision_extract_and_flatten`` -> ``_encode_messages`` ->
             ``_maybe_canonicalize_committed_reasoning`` (the committed-think
             substitution and the committed-id splice), on the tokenizer the
             runtime loads (``mtplx.runtime._load_tokenizer_resilient``).
             tests/test_agent_prompt_fidelity.py proves this chain returns the
             ids the endpoint hands to generation.

Two walks over each synthetic session:

  stateless  every turn boundary rendered on its own (a cold request). Also
             checks the cache side of the same seam: the prefix the postcommit
             banks after an assistant turn must be a prefix of the next
             request (``postcommit_prefix_ok``).
  session    the same walk with a committed stream between turns: turn k's
             served prompt plus the ids the model "generated" for the next
             assistant message (canonical; with BPE seams split or joined the
             way the tokenizer never produces; with the end-of-turn token
             kept; with a trailing newline the response stripped). Reports
             what the next prompt's ids are, whether they extend the committed
             stream, and how they differ from the reference.

Every difference is decoded and attributed to a documented mechanism
(file:line in ``MECHANISMS``) or reported as UNEXPLAINED. Output: one JSON
line per (session, turn), a summary table on stderr, exit code 1 when anything
is unexplained.

All conversations are synthetic and written in this file. Nothing is read
from a request log, a session bank or a client's history.

Usage:
  .venv/bin/python scripts/audit_agent_prompt_fidelity.py \
      [--pack DIR ...] [--scenario stateless|session] [--only-session NAME] \
      [--out results.jsonl] [--summary-json summary.json]
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PACKS = (
    "Qwen3.8-Flash-Next-MTPLX-Optimized-Speed",
    "Qwen3.8-27B-MTPLX-Optimized-Speed",
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_SEAM = "<|im_start|>assistant\n<think>\n"

# ---------------------------------------------------------------------------
# Documented mechanisms a difference may be attributed to. ``intended`` False
# marks a mechanism that explains the bytes but whose effect on the model is
# the owner's call (it is still reported as explained, with the note).
# ---------------------------------------------------------------------------
MECHANISMS: dict[str, dict[str, Any]] = {
    "tool_contract_system_injection": {
        "where": "mtplx/server/openai.py:8124 _with_mtplx_tool_contract, text at :7557 "
        "_mtplx_tool_contract_text; mode chosen at :21107 _tool_prompt_mode_for_request "
        "(pi/hermes -> hybrid, opencode -> compact, no client header -> launch mode)",
        "what": "MTPLX appends its tool contract (dated, one-line tool signatures, format "
        "example, steering clauses) to the first system message in the hybrid and compact "
        "tool-prompt modes; it creates the system message when the client sent none.",
    },
    "compact_mode_template_tools_omitted": {
        "where": "mtplx/server/openai.py:6737 _template_tools_for_prompt_mode; mode at :21107 "
        "(client opencode -> compact)",
        "what": "Compact mode passes no tools to the chat template: the template's '# Tools' "
        "block with the JSON schemas and descriptions is not rendered; the contract's one-line "
        "signatures stand in for it.",
    },
    "tool_schema_key_sort": {
        "where": "mtplx/server/openai.py:8714 _normalize_tool_specs",
        "what": "Every tool schema is re-serialized with sorted object keys so a client's JSON "
        "key order cannot change the system prefix. The JSON values are equal; the key order "
        "the model reads is alphabetical instead of the client's (type, function, name, "
        "description, parameters).",
    },
    "generation_seam_segmentation": {
        "where": "mtplx/server/openai.py:13304 _qwen_assistant_generation_boundaries, :13331 "
        "_assistant_generation_boundaries, used at :14226 "
        "_encode_generation_compatible_tool_history and in :14530 _encode_messages_uncached",
        "what": "Thinking on: history is encoded in segments cut right after every "
        "'<|im_start|>assistant\\n<think>\\n', so the ids are the ones the model saw when it "
        "generated that turn (its prompt ended there). Same text; the only token a one-pass "
        "encode merges across the cut is the blank line of an EMPTY think block ('\\n\\n' "
        "stays '\\n','\\n').",
    },
    "opencode_tool_call_preamble_strip": {
        "where": "mtplx/server/openai.py:12806 in _canonicalize_agent_transcript "
        "(strip_tool_call_preamble_text, set for OpenCode at mtplx/server/request_policy.py:592)",
        "what": "For OpenCode the visible text of an assistant turn that also made tool calls "
        "is removed from history. With a live session and thinking on, the committed turn "
        "body puts it back (:13757 _substitute_committed_reasoning_messages).",
    },
    "thinking_off_history_reasoning_dropped": {
        "where": "mtplx/server/openai.py:14562 include_reasoning (_encode_messages_uncached) "
        "and :13073 _message_to_template_dict",
        "what": "With thinking off, reasoning_content echoed on history turns is not given to "
        "the template: those turns render the empty think scaffold.",
    },
    "consecutive_role_merge": {
        "where": "mtplx/server/omlx_bridge/adapter.py:134 _merge_consecutive_roles",
        "what": "Two consecutive user (or plain assistant) messages are joined with a blank "
        "line into one turn.",
    },
    "committed_id_splice": {
        "where": "mtplx/server/openai.py:13408 _splice_committed_token_ids, called from :13893 "
        "_maybe_canonicalize_committed_reasoning",
        "what": "Where the re-rendered history and the session's committed stream decode to "
        "the same text, the ids the model itself generated are served (same text, different "
        "token boundaries).",
    },
    "committed_whitespace_restore": {
        "where": "mtplx/server/openai.py:13465 (whitespace branch of _splice_committed_token_ids)",
        "what": "Whitespace tokens the model generated and the response stripped are put back "
        "from the committed stream.",
    },
}


# ---------------------------------------------------------------------------
# Synthetic conversations
# ---------------------------------------------------------------------------
def _fn(name: str, description: str, properties: dict[str, Any], required: list[str], **extra: Any) -> dict[str, Any]:
    parameters: dict[str, Any] = {"type": "object", "properties": properties, "required": required}
    parameters.update(extra)
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def pi_tools() -> list[dict[str, Any]]:
    """Eight tools, Pi-shaped: nested objects, arrays, enums, required."""
    return [
        _fn("read", "Read a file. Returns numbered lines.", {
            "path": {"type": "string", "description": "Path relative to the working directory"},
            "offset": {"type": "integer", "description": "First line (1-based)", "minimum": 1},
            "limit": {"type": "integer", "description": "Maximum number of lines"},
        }, ["path"]),
        _fn("write", "Create or overwrite a file with the given content.", {
            "path": {"type": "string"},
            "content": {"type": "string", "description": "Full file content"},
        }, ["path", "content"]),
        _fn("edit", "Apply exact-text replacements to one file.", {
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {"type": "string"},
                        "newText": {"type": "string"},
                        "occurrence": {"type": "string", "enum": ["first", "last", "all"], "default": "first"},
                    },
                    "required": ["oldText", "newText"],
                },
            },
        }, ["path", "edits"]),
        _fn("bash", "Run a shell command in the working directory.", {
            "command": {"type": "string"},
            "timeout": {"type": "number", "description": "Seconds", "default": 120},
            "env": {"type": "object", "additionalProperties": {"type": "string"}},
        }, ["command"]),
        _fn("grep", "Search file contents with a regular expression.", {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
            "ignoreCase": {"type": "boolean", "default": False},
            "context": {"type": "integer", "minimum": 0, "maximum": 20},
            "outputMode": {"type": "string", "enum": ["content", "files", "count"]},
        }, ["pattern"]),
        _fn("find", "Find files by glob pattern.", {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "type": {"type": "string", "enum": ["file", "directory", "any"]},
        }, ["pattern"]),
        _fn("ls", "List a directory.", {
            "path": {"type": "string"},
            "depth": {"type": "integer", "minimum": 1, "maximum": 5},
        }, []),
        _fn("todo_update", "Replace the task list.", {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
                        "priority": {"type": "string", "enum": ["low", "normal", "high"]},
                        "meta": {
                            "type": "object",
                            "properties": {
                                "owner": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["owner"],
                        },
                    },
                    "required": ["id", "title", "status"],
                },
            },
        }, ["items"]),
    ]


def opencode_tools() -> list[dict[str, Any]]:
    """OpenCode-shaped tool list (zod-style schemas: $schema, additionalProperties)."""
    def oc(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return _fn(name, description, properties, required, additionalProperties=False,
                   **{"$schema": "http://json-schema.org/draft-07/schema#"})
    return [
        oc("bash", "Executes a given bash command in a persistent shell session.", {
            "command": {"type": "string", "description": "The command to execute"},
            "timeout": {"type": "number", "description": "Optional timeout in milliseconds"},
            "description": {"type": "string", "description": "Clear, concise description of what this command does in 5-10 words."},
        }, ["command", "description"]),
        oc("read", "Reads a file from the local filesystem.", {
            "filePath": {"type": "string", "description": "The path to the file to read"},
            "offset": {"type": "number"},
            "limit": {"type": "number"},
        }, ["filePath"]),
        oc("write", "Writes a file to the local filesystem.", {
            "filePath": {"type": "string"},
            "content": {"type": "string"},
        }, ["filePath", "content"]),
        oc("edit", "Performs exact string replacements in files.", {
            "filePath": {"type": "string"},
            "oldString": {"type": "string"},
            "newString": {"type": "string"},
            "replaceAll": {"type": "boolean"},
        }, ["filePath", "oldString", "newString"]),
        oc("glob", "Fast file pattern matching tool.", {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
        }, ["pattern"]),
        oc("grep", "Fast content search tool.", {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "include": {"type": "string"},
        }, ["pattern"]),
        oc("todowrite", "Use this tool to create and manage a structured task list.", {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        "id": {"type": "string"},
                    },
                    "required": ["content", "status", "priority", "id"],
                    "additionalProperties": False,
                },
            },
        }, ["todos"]),
        oc("webfetch", "Fetches content from a specified URL.", {
            "url": {"type": "string"},
            "format": {"type": "string", "enum": ["text", "markdown", "html"]},
            "timeout": {"type": "number"},
        }, ["url", "format"]),
        oc("task", "Launch a new agent to handle complex, multistep tasks autonomously.", {
            "description": {"type": "string"},
            "prompt": {"type": "string"},
            "subagent_type": {"type": "string"},
        }, ["description", "prompt", "subagent_type"]),
    ]


def hermes_tools() -> list[dict[str, Any]]:
    return [
        _fn("terminal", "Run a command in the user's terminal and return its output.", {
            "command": {"type": "string"},
            "background": {"type": "boolean", "default": False},
        }, ["command"]),
        _fn("write_file", "Write text to a file, replacing it.", {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, ["path", "content"]),
        _fn("read_file", "Read a text file.", {
            "path": {"type": "string"},
        }, ["path"]),
    ]


class _Builder:
    def __init__(self, system: str | None) -> None:
        self.messages: list[dict[str, Any]] = []
        self._call = 0
        if system is not None:
            self.messages.append({"role": "system", "content": system})

    def user(self, text: str) -> "_Builder":
        self.messages.append({"role": "user", "content": text})
        return self

    def assistant(self, content: str | None = "", *, reasoning: str | None = None,
                  calls: Sequence[tuple[str, dict[str, Any]]] = (), args_as_string: bool = True) -> "_Builder":
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        if calls:
            tool_calls = []
            for name, arguments in calls:
                self._call += 1
                tool_calls.append({
                    "id": f"call_{self._call:04d}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False) if args_as_string else arguments,
                    },
                })
            message["tool_calls"] = tool_calls
        self.messages.append(message)
        return self

    def tool(self, content: str, *, back: int = 0) -> "_Builder":
        """Result for the most recent unanswered call (``back`` counts from the newest)."""
        call_id = None
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for message in reversed(self.messages):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                pending = [c["id"] for c in message["tool_calls"] if c["id"] not in answered]
                if pending:
                    call_id = pending[0]
                break
        self.messages.append({"role": "tool", "tool_call_id": call_id or f"call_{self._call:04d}", "content": content})
        return self


def _numbered_file(lines: int) -> str:
    rows = []
    for i in range(1, lines + 1):
        if i % 17 == 0:
            rows.append(f"{i:>5}\t")
        elif i % 5 == 0:
            rows.append(f"{i:>5}\t    return compute_total(items[{i}], rate={i / 10:.1f})  # tier {i % 7}")
        else:
            rows.append(f"{i:>5}\tdef handler_{i}(request, *, retries={i % 4}):")
    return "\n".join(rows)


PI_SYSTEM = (
    "You are an expert coding assistant operating inside a terminal harness.\n"
    "You help the user by reading files, executing commands, editing code and writing new files.\n\n"
    "Available tools:\n- read: Read file contents\n- bash: Execute bash commands\n- edit: Make surgical edits\n"
    "- write: Create or overwrite files\n\nGuidelines:\n- Use bash for file operations like ls, grep, find\n"
    "- Use read to examine files before editing\n- Be concise in your responses\n- Show file paths clearly\n\n"
    "Current date and time: Friday, September 18, 2026 at 09:00:00 AM\nCurrent working directory: /work/ledger"
)

SPECIAL_LOOKING = (
    "src/render.py:41:    END = \"<|im_end|>\"  # chat end marker\n"
    "src/render.py:42:    THINK_CLOSE = \"</think>\"\n"
    "src/render.py:43:    start = \"<|im_start|>assistant\"\n"
    "src/render.py:57:    if text.endswith(\"</tool_response>\") or \"<tool_call>\" in text:\n"
    "src/render.py:58:        return text.replace(\"<|endoftext|>\", \"\")\n"
    "docs/format.md:9:<think>\nnot a real think block\n</think>"
)


def session_pi_60() -> dict[str, Any]:
    b = _Builder(PI_SYSTEM)
    b.user("The invoice totals are off by a cent on some orders. Find out why and fix it.")
    b.assistant("", reasoning="The user reports a rounding problem. I should look at the project layout first, then find where totals are computed.",
                calls=[("ls", {"path": ".", "depth": 2})])
    b.tool("ledger/\n  __init__.py\n  totals.py\n  money.py\n  io.py\ntests/\n  test_totals.py\n  test_money.py\npyproject.toml\nREADME.md")
    b.assistant("", reasoning="Two candidates: totals.py and money.py. Read both in one go.",
                calls=[("read", {"path": "ledger/totals.py"}), ("read", {"path": "ledger/money.py", "offset": 1, "limit": 120})])
    b.tool(_numbered_file(140))
    b.tool(json.dumps({"path": "ledger/money.py", "lines": 3, "content": ["from decimal import Decimal", "", "CENT = Decimal('0.01')"], "truncated": False}, indent=2))
    b.assistant("I'll run the money tests to see the failure.", reasoning="money.py uses Decimal but totals.py multiplies floats. Confirm with the tests before editing.",
                calls=[("bash", {"command": "python -m pytest tests/test_money.py -q", "timeout": 60})])
    b.tool("")
    b.user("Heads up: do not touch the public function names, other services import them.")
    b.assistant("", reasoning="Constraint noted: keep names. The empty test output means the command printed nothing; rerun with -rA and search for float usage.",
                calls=[("grep", {"pattern": "float\\(|round\\(", "path": "ledger", "outputMode": "content", "context": 1}),
                       ("bash", {"command": "python -m pytest tests -q -rA 2>&1 | tail -n 20", "env": {"PYTHONHASHSEED": "0"}})])
    b.tool("ledger/totals.py:12:    subtotal = float(price) * qty\nledger/totals.py:19:    return round(subtotal * (1 + rate), 2)")
    b.tool("F.                                                                       [100%]\nFAILED tests/test_totals.py::test_cent_rounding - assert 10.07 == Decimal('10.08')\n1 failed, 1 passed in 0.04s")
    b.assistant("", reasoning="Binary float: 10.075 rounds down. Replace with Decimal quantize ROUND_HALF_UP. One edit call, two replacements.",
                calls=[("edit", {"path": "ledger/totals.py", "edits": [
                    {"oldText": "    subtotal = float(price) * qty", "newText": "    subtotal = Decimal(str(price)) * qty"},
                    {"oldText": "    return round(subtotal * (1 + rate), 2)", "newText": "    return (subtotal * (1 + Decimal(str(rate)))).quantize(CENT, rounding=ROUND_HALF_UP)", "occurrence": "first"},
                ]})])
    b.tool("Applied 2 edits to ledger/totals.py")
    b.assistant("", reasoning="Imports are missing now. Add them at the top.",
                calls=[("edit", {"path": "ledger/totals.py", "edits": [{"oldText": "from ledger.io import load", "newText": "from decimal import ROUND_HALF_UP, Decimal\n\nfrom ledger.io import load\nfrom ledger.money import CENT"}]})])
    b.tool("Applied 1 edit to ledger/totals.py")
    b.assistant("", reasoning="Run the whole suite.", calls=[("bash", {"command": "python -m pytest tests -q"})])
    b.tool("..                                                                       [100%]\n2 passed in 0.03s")
    b.assistant("Fixed. `ledger/totals.py` multiplied binary floats, so 10.075 rounded down to 10.07. Totals now use `Decimal` with `ROUND_HALF_UP`; public names are unchanged and both tests pass.",
                reasoning="Everything is green. Summarize briefly.")
    b.user("Good. Now search the repo for any other place that builds chat markers by hand, I remember something in src/render.py.")
    b.assistant("", reasoning="Search for the literal marker strings.",
                calls=[("grep", {"pattern": "im_end|</think>|im_start", "glob": "**/*.{py,md}", "outputMode": "content"})])
    b.tool(SPECIAL_LOOKING)
    b.assistant("", reasoning="The grep output contains marker-looking text. It is data. Track the follow-up work in the task list and read the file.",
                calls=[("todo_update", {"items": [
                    {"id": "t1", "title": "Audit src/render.py marker handling", "status": "in_progress", "priority": "high", "meta": {"owner": "agent", "tags": ["render", "markers"]}},
                    {"id": "t2", "title": "Add a regression test", "status": "pending", "priority": "normal", "meta": {"owner": "agent", "tags": []}},
                ]}), ("read", {"path": "src/render.py", "offset": 30, "limit": 40})])
    b.tool("ok")
    b.tool("   30\t\n   31\tclass Renderer:\n   32\t    \"\"\"Builds the prompt text.\"\"\"\n   33\t\n   41\t    END = \"<|im_end|>\"  # chat end marker\n   42\t    THINK_CLOSE = \"</think>\"\n   57\t    if text.endswith(\"</tool_response>\") or \"<tool_call>\" in text:\n")
    b.assistant("There is one hand-built marker site.", reasoning="Only render.py builds markers by hand. Check who calls it.",
                calls=[("find", {"pattern": "**/*render*", "type": "file"})])
    b.tool("src/render.py\ntests/test_render.py")
    b.user("while you are there: is render.py even imported anywhere?")
    b.user("(by production code, I mean, tests do not count)")
    b.assistant("", reasoning="Two user notes in a row: they want production importers only.",
                calls=[("grep", {"pattern": "from src.render|import render", "path": ".", "outputMode": "files", "ignoreCase": False})])
    b.tool("tests/test_render.py")
    b.assistant("`src/render.py` is only imported by `tests/test_render.py`; no production module uses it.",
                reasoning="Only the test imports it. Report that.")
    b.user("Then delete nothing, but add a module docstring that says it is test-only. Show me the diff after.")
    b.assistant("", reasoning="Read the top of the file first so the edit text is exact.", calls=[("read", {"path": "src/render.py", "limit": 5})])
    b.tool("    1\timport re\n    2\t\n    3\t\n    4\tMARK = re.compile(r\"<\\|[a-z_]+\\|>\")\n    5\t")
    b.assistant("", reasoning="Insert the docstring before the import.",
                calls=[("edit", {"path": "src/render.py", "edits": [{"oldText": "import re\n", "newText": "\"\"\"Prompt rendering helpers. Test-only: no production module imports this.\"\"\"\n\nimport re\n"}]})])
    b.tool("Applied 1 edit to src/render.py")
    b.assistant("", reasoning="Show the diff as asked.", calls=[("bash", {"command": "git diff --stat && git diff src/render.py"})])
    b.tool(" src/render.py | 2 ++\n 1 file changed, 2 insertions(+)\ndiff --git a/src/render.py b/src/render.py\n--- a/src/render.py\n+++ b/src/render.py\n@@ -1,3 +1,5 @@\n+\"\"\"Prompt rendering helpers. Test-only: no production module imports this.\"\"\"\n+\n import re\n ")
    b.assistant("Done. The diff adds a two-line module docstring at the top of `src/render.py` and changes nothing else.",
                reasoning="The diff is what was asked for.")
    b.user("Mark the tasks done and give me a one-line summary of the session.")
    b.assistant("", reasoning="Close both tasks.",
                calls=[("todo_update", {"items": [
                    {"id": "t1", "title": "Audit src/render.py marker handling", "status": "done", "priority": "high", "meta": {"owner": "agent", "tags": ["render", "markers"]}},
                    {"id": "t2", "title": "Add a regression test", "status": "done", "priority": "normal", "meta": {"owner": "agent"}},
                ]})])
    b.tool("ok")
    b.assistant("Fixed cent rounding in `ledger/totals.py` with `Decimal`, and documented `src/render.py` as test-only.",
                reasoning="One line.")
    b.user("One more thing, list the tests directory and tell me how many test files there are.")
    b.assistant("", reasoning="List it.", calls=[("ls", {})])
    b.tool("test_money.py\ntest_render.py\ntest_totals.py")
    b.assistant("Three test files: `test_money.py`, `test_render.py`, `test_totals.py`.", reasoning="Three entries.")
    b.user("Run them one final time with verbose output.")
    b.assistant("", reasoning="Verbose run.", calls=[("bash", {"command": "python -m pytest tests -v", "timeout": 120})])
    b.tool("tests/test_money.py::test_cent PASSED\ntests/test_render.py::test_markers PASSED\ntests/test_totals.py::test_cent_rounding PASSED\n3 passed in 0.05s")
    b.assistant("All three tests pass.", reasoning="Green.")
    b.user("Thanks. Anything you would still change?")
    b.assistant("", reasoning="Check the README mentions Decimal before answering.", calls=[("grep", {"pattern": "Decimal|float", "path": "README.md", "outputMode": "count"})])
    b.tool("0\n\n--- full listing follows (long tool result, passed through whole) ---\n" + _numbered_file(600))
    b.assistant("I would add one README line saying amounts are `Decimal`, never float. Nothing else.", reasoning="README is silent on it; suggest one line.")
    b.user("Do it.")
    b.assistant("Adding the line.", reasoning="Append one sentence under the first heading.",
                calls=[("edit", {"path": "README.md", "edits": [{"oldText": "# ledger\n", "newText": "# ledger\n\nAmounts are `Decimal`, never float.\n", "occurrence": "first"}]})])
    b.tool("Applied 1 edit to README.md")
    assert len(b.messages) == 60, len(b.messages)
    return {"name": "pi_60", "shape": "pi", "tools": pi_tools(), "messages": b.messages}


OPENCODE_SYSTEM = (
    "You are opencode, an interactive CLI tool that helps users with software engineering tasks.\n\n"
    "# Tone and style\nYou should be concise, direct, and to the point.\n\n"
    "# Tool usage policy\n- When doing file search, prefer to use the task tool.\n"
    "- You can call multiple tools in a single response.\n\n"
    "Here is some useful information about the environment you are running in:\n<env>\n"
    "  Working directory: /work/notes-app\n  Is directory a git repo: yes\n  Platform: darwin\n  Today's date: Fri Sep 18 2026\n</env>"
)


def session_opencode() -> dict[str, Any]:
    b = _Builder(OPENCODE_SYSTEM)
    b.user("Add a --json flag to the export command.")
    b.assistant("I'll look at the export command first.", reasoning="Find the command definition.",
                calls=[("glob", {"pattern": "src/**/export*.ts"})])
    b.tool("/work/notes-app/src/commands/export.ts")
    b.assistant("", reasoning="Read it.", calls=[("read", {"filePath": "/work/notes-app/src/commands/export.ts"})])
    b.tool("<file>\n00001| import { Command } from \"commander\"\n00002| \n00003| export const exportCmd = new Command(\"export\")\n00004|   .option(\"--out <file>\")\n00005|   .action(async (opts) => {\n00006|     await run(opts)\n00007|   })\n</file>")
    b.assistant("Adding the flag now.", reasoning="Insert the option after --out and thread it to run().",
                calls=[("edit", {"filePath": "/work/notes-app/src/commands/export.ts", "oldString": "  .option(\"--out <file>\")\n", "newString": "  .option(\"--out <file>\")\n  .option(\"--json\", \"emit JSON\")\n"}),
                       ("todowrite", {"todos": [{"content": "Thread --json through run()", "status": "in_progress", "priority": "high", "id": "1"}]})])
    b.tool("Edit applied successfully.")
    b.tool("[{\"content\":\"Thread --json through run()\",\"status\":\"in_progress\",\"priority\":\"high\",\"id\":\"1\"}]")
    # OpenCode does not always echo reasoning: one turn without it.
    b.assistant("", calls=[("bash", {"command": "bun test export", "description": "Run export tests"})])
    b.tool("bun test v1.2.0\n\n 3 pass\n 0 fail\nRan 3 tests across 1 file. [41.00ms]")
    b.assistant("The `--json` flag is in place and the export tests pass.", reasoning="Done; report.")
    b.user("Now document it in the README under Usage.")
    b.assistant("", reasoning="Find the Usage section.", calls=[("grep", {"pattern": "^## Usage", "include": "README.md"})])
    b.tool("Found 1 matches\n/work/notes-app/README.md:\n  Line 12: ## Usage")
    b.assistant("", reasoning="Read around it.", calls=[("read", {"filePath": "/work/notes-app/README.md", "offset": 10, "limit": 12})])
    b.tool("<file>\n00010| \n00011| \n00012| ## Usage\n00013| \n00014| ```sh\n00015| notes export --out notes.md\n00016| ```\n</file>")
    b.assistant("", reasoning="Append a JSON example.",
                calls=[("edit", {"filePath": "/work/notes-app/README.md", "oldString": "notes export --out notes.md\n```", "newString": "notes export --out notes.md\nnotes export --json > notes.json\n```", "replaceAll": False})])
    b.tool("Edit applied successfully.")
    b.assistant("Documented under Usage.", reasoning="Done.")
    b.user("thanks, what did you change overall?")
    return {"name": "opencode_22", "shape": "opencode", "tools": opencode_tools(), "messages": b.messages}


HERMES_SYSTEM = (
    "You are Hermes, a terminal agent. Carry out the request with the available tools and report the result.\n"
    "Conversation started: Friday, September 18, 2026"
)


def session_hermes_oneshot() -> dict[str, Any]:
    b = _Builder(HERMES_SYSTEM)
    b.user("Create hello.py that prints the current ISO week number, then run it.")
    return {"name": "hermes_oneshot", "shape": "hermes", "tools": hermes_tools(), "messages": b.messages}


def session_no_system() -> dict[str, Any]:
    """Tools and no system message: the server has to create the system turn
    its contract lives in; the template creates one for the tools block."""
    b = _Builder(None)
    b.user("How many lines does notes.txt have?")
    b.assistant("", reasoning="Count with wc.", calls=[("terminal", {"command": "wc -l notes.txt"})])
    b.tool("      12 notes.txt")
    b.assistant("12 lines.", reasoning="Report the number.")
    b.user("And words?")
    return {"name": "no_system", "shape": "hermes", "tools": hermes_tools(), "messages": b.messages}


def session_hermes_loop() -> dict[str, Any]:
    b = _Builder(HERMES_SYSTEM)
    b.user("Create hello.py that prints the current ISO week number, then run it.")
    b.assistant("", calls=[("write_file", {"path": "hello.py", "content": "import datetime\n\nprint(datetime.date.today().isocalendar().week)\n"})])
    b.tool("wrote 72 bytes")
    b.assistant(None, calls=[("terminal", {"command": "python hello.py", "background": False})])
    b.tool("38\n")
    b.assistant("`hello.py` prints the ISO week; today it prints 38.")
    b.user("Make it print the year too.")
    return {"name": "hermes_loop", "shape": "hermes", "tools": hermes_tools(), "messages": b.messages}


def session_multilingual() -> dict[str, Any]:
    b = _Builder("أنت مساعد برمجة. 你是一个编程助手。 Keep answers short. \U0001F9D1‍\U0001F4BB")
    b.user("اِقْرَأْ مَلَفَّ الإعدادات ثم لخّصه. 設定ファイルを読んで要約してください。 \U0001F64F\U0001F3FD")
    b.assistant("", reasoning="المستخدم يريد ملخصًا لملف الإعدادات. まず読む。先读文件。",
                calls=[("read", {"path": "config/الإعدادات.yaml"})])
    b.tool("    1\tلغة: العربية\n    2\t名前: 設定\n    3\t이름: 설정\n    4\tभाषा: हिन्दी\n    5\tภาษา: ไทย\n    6\temoji: \U0001F1EF\U0001F1F5 \U0001F468‍\U0001F469‍\U0001F467 ❤️\n    7\tcafé: naïve — “quotes” … ½")
    b.assistant("الملف يحدّد اللغة (العربية) والاسم (設定 / 설정) ويحتوي على رموز تعبيرية \U0001F1EF\U0001F1F5. 要約：言語と名前の設定です。",
                reasoning="ملخص قصير بلغتين. हिन्दी और ไทย भी मौजूद हैं।")
    b.user("اكتب الملخص في ملف 要約.md 并加上一个表情 \U0001F389")
    b.assistant("سأكتب الملف الآن.", reasoning="Write the file with mixed scripts.",
                calls=[("write", {"path": "要約.md", "content": "# ملخّص / 要約 / 요약\n\n- لغة: العربية\n- 名前: 設定\n- भाषा: हिन्दी\n- ภาษา: ไทย\n\n\U0001F389\n"})])
    b.tool("✓ كُتب الملف (٧ أسطر)")
    b.assistant("تمّ. 完了しました。 \U0001F389", reasoning="Done.")
    b.user("ممتاز، شكرًا! 谢谢 \U0001F600")
    return {"name": "multilingual", "shape": "pi", "tools": pi_tools(), "messages": b.messages}


ODD_CODE = (
    "def f(x):\r\n"
    "\tif x:   \r\n"
    "\t\treturn\t1  \r\n"
    "    \r\n"
    "\t# mixed\t tabs and   spaces nbsp\r\n"
    "\treturn 0\x0c\r\n"
    "\r\n"
    "\r\n"
    "class  A :\n"
    "  \tdef g( self ) :\n"
    "\t  \tpass   \n"
    "   \n"
    "\n"
    "x = 'no newline at end'"
)


def session_odd_whitespace() -> dict[str, Any]:
    b = _Builder("You are a careful editor. Preserve whitespace exactly.  \n\n")
    b.user("  \n\tShow me legacy.py exactly as it is on disk.\n\n")
    b.assistant("", reasoning="Read it without changing anything.", calls=[("read", {"path": "legacy.py"})])
    b.tool("\n\n" + ODD_CODE + "\n\n\n")
    b.assistant("\n\nHere it is, CRLF and tabs intact:\n\n```python\n" + ODD_CODE + "\n```\n\n",
                reasoning="  The file mixes CRLF and LF, tabs and spaces.\n\tKeep them.  \n")
    b.user("Write it back with a trailing newline, and also write a second file that starts with two blank lines.")
    b.assistant("", reasoning="Two writes in one turn.",
                calls=[("write", {"path": "legacy.py", "content": ODD_CODE + "\n"}),
                       ("write", {"path": "blank_first.txt", "content": "\n\nstarts after two blank lines  \n\ttab line\n\n"})])
    b.tool("wrote legacy.py")
    b.tool("   \n")
    b.assistant("Both files are written.   ", reasoning="Done.")
    b.user("Now replace the form feed with nothing.")
    b.assistant("", reasoning="Exact-text edit of the form feed line.",
                calls=[("edit", {"path": "legacy.py", "edits": [{"oldText": "\treturn 0\x0c\r\n", "newText": "\treturn 0\r\n"}]})])
    b.tool("Applied 1 edit to legacy.py\r\n")
    b.messages.append({"role": "user", "content": [
        {"type": "text", "text": "and show the first three lines\n"},
        {"type": "text", "text": "\t(keep the CRLF)  "},
    ]})
    return {"name": "odd_whitespace_code", "shape": "pi", "tools": pi_tools(), "messages": b.messages}


def session_plain_chat() -> dict[str, Any]:
    """No tools at all: the baseline where nothing but the template speaks."""
    b = _Builder("You are a concise assistant.")
    b.user("What does HTTP status 409 mean?")
    b.assistant("409 Conflict: the request clashes with the current state of the resource, for example an edit based on a stale version.",
                reasoning="A short definition with one example is enough.")
    b.user("And 412?")
    b.assistant("412 Precondition Failed: a condition in the request headers, such as `If-Match`, evaluated to false on the server.",
                reasoning="Same shape as the previous answer.")
    b.user("Which one should an optimistic-locking API return?")
    b.assistant("Use 412 when the client sent `If-Match` and the version differs; use 409 when there is no precondition header and the conflict is found in the body.",
                reasoning="Distinguish by whether a precondition header was sent.")
    b.user("Thanks. One-line summary?")
    return {"name": "plain_chat", "shape": "plain", "tools": None, "messages": b.messages}


def all_sessions() -> list[dict[str, Any]]:
    return [
        session_plain_chat(),
        session_pi_60(),
        session_opencode(),
        session_hermes_oneshot(),
        session_hermes_loop(),
        session_no_system(),
        session_multilingual(),
        session_odd_whitespace(),
    ]


# Client lanes: the headers each real client sends (mtplx/pi.py:350,
# mtplx/opencode.py:94, mtplx/commands/public.py:11985) plus two controls.
LANES: dict[str, dict[str, Any]] = {
    "pi": {"headers": {"x-mtplx-client": "pi", "x-mtplx-session-id": "audit-pi"}},
    "opencode": {"headers": {"x-mtplx-client": "opencode", "x-mtplx-session-id": "audit-opencode"}},
    "hermes": {"headers": {"x-mtplx-client": "hermes", "x-mtplx-session-id": "audit-hermes"}},
    # No client header: an anonymous OpenAI-API caller (launch tool-prompt mode).
    "anonymous": {"headers": {}},
    # Control: the same anonymous caller asking for the template-native mode.
    "native": {"headers": {"x-mtplx-tool-prompt-mode": "native"}},
}

SHAPE_LANES = {
    "plain": ("pi", "opencode", "hermes", "anonymous"),
    "pi": ("pi", "anonymous", "native"),
    "opencode": ("opencode", "native"),
    "hermes": ("hermes", "native"),
}


# label -> (enable_thinking, history carries reasoning_content)
# off            a session that ran with thinking off throughout (no reasoning
#                was ever generated, so none is echoed)
# off_after_on   the history was generated with thinking on and the user then
#                switched thinking off
THINKING_MODES: dict[str, tuple[bool, bool]] = {
    "on": (True, True),
    "off": (False, False),
    "off_after_on": (False, True),
}


def without_reasoning(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]


def turn_boundaries(messages: Sequence[dict[str, Any]]) -> list[int]:
    """Prefix lengths at which a client sends a request: before every
    assistant message, and the whole conversation when it ends on a user or
    tool message."""
    cuts = [i for i, m in enumerate(messages) if m["role"] == "assistant" and i > 0]
    if messages and messages[-1]["role"] in {"user", "tool"}:
        cuts.append(len(messages))
    return [c for c in cuts if any(m["role"] == "user" for m in messages[:c])]


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
def _split_pattern(pre: Any) -> str | None:
    if isinstance(pre, dict):
        pattern = pre.get("pattern")
        if isinstance(pattern, dict) and isinstance(pattern.get("Regex"), str):
            return pattern["Regex"]
        for value in pre.values():
            found = _split_pattern(value)
            if found:
                return found
    elif isinstance(pre, list):
        for value in pre:
            found = _split_pattern(value)
            if found:
                return found
    return None


class Reference:
    """The pack's own template + tokenizer, with no MTPLX code in the path."""

    def __init__(self, pack: Path) -> None:
        from tokenizers import Regex, Tokenizer, pre_tokenizers
        from transformers import AutoTokenizer

        self.pack = pack
        tokenizer_config = json.loads((pack / "tokenizer_config.json").read_text(encoding="utf-8"))
        sidecar = pack / "chat_template.jinja"
        self.template = (
            sidecar.read_text(encoding="utf-8") if sidecar.exists() else tokenizer_config.get("chat_template")
        )
        if not self.template:
            raise RuntimeError(f"{pack} ships no chat template")
        self.hf = AutoTokenizer.from_pretrained(str(pack))
        self.hf.chat_template = self.template
        self.encoder = Tokenizer.from_file(str(pack / "tokenizer.json"))
        file_json = json.loads((pack / "tokenizer.json").read_text(encoding="utf-8"))
        file_regex = _split_pattern(file_json.get("pre_tokenizer"))
        declared = tokenizer_config.get("pretokenize_regex")
        self.regex_source = "tokenizer.json"
        if isinstance(declared, str) and declared and declared != file_regex:
            # The pack declares the regex the model was trained with; its
            # tokenizer.json was re-saved with another one.
            self.encoder.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(Regex(declared), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
            ])
            self.regex_source = "tokenizer_config.json:pretokenize_regex"
        self.vocab = file_json["model"]["vocab"]
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merge_parts: dict[str, tuple[str, str]] = {}
        for merge in file_json["model"]["merges"]:
            left, right = merge if isinstance(merge, (list, tuple)) else merge.split(" ", 1)
            self.merge_parts.setdefault(left + right, (left, right))
        self.im_start_id = self.encoder.token_to_id(IM_START)
        self.control_ids = {
            self.encoder.token_to_id(t)
            for t in (IM_START, IM_END, "<think>", "</think>", "<tool_call>", "</tool_call>")
        }

    @staticmethod
    def template_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """OpenAI wire shape -> what the template reads. The only change is
        the protocol one every reference stack makes: tool-call arguments
        arrive as a JSON string and the template iterates them as a mapping."""
        out: list[dict[str, Any]] = []
        for message in messages:
            item = dict(message)
            if item.get("tool_calls"):
                calls = []
                for call in item["tool_calls"]:
                    call = json.loads(json.dumps(call))
                    function = call.get("function") or {}
                    if isinstance(function.get("arguments"), str):
                        function["arguments"] = json.loads(function["arguments"] or "{}")
                    calls.append(call)
                item["tool_calls"] = calls
            out.append(item)
        return out

    def render(self, messages: Sequence[dict[str, Any]], tools: list[dict[str, Any]] | None, *,
               enable_thinking: bool, reasoning_effort: str | None, preserve_thinking: bool | None = True,
               add_generation_prompt: bool = True) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
        if preserve_thinking is not None:
            kwargs["preserve_thinking"] = preserve_thinking
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs["tools"] = tools
        return self.hf.apply_chat_template(self.template_messages(messages), **kwargs)

    def encode(self, text: str) -> list[int]:
        return list(self.encoder.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: Iterable[int]) -> str:
        return self.encoder.decode([int(i) for i in ids], skip_special_tokens=False)


# ---------------------------------------------------------------------------
# MTPLX request path
# ---------------------------------------------------------------------------
class _AuditSessions:
    """The two session calls the prompt path makes, over a plain dict."""

    def __init__(self) -> None:
        self.committed: dict[str, tuple[int, ...]] = {}

    def resolve_session_id(self, *, headers: dict[str, str], **_kwargs: Any) -> tuple[str | None, str | None]:
        session_id = headers.get("x-mtplx-session-id")
        return (session_id, "header") if session_id else (None, None)

    def peek(self, session_id: str) -> Any:
        if session_id not in self.committed:
            return None
        return SimpleNamespace(committed_token_ids=self.committed[session_id])


class MtplxPath:
    """The server's prompt path over a state that holds no model."""

    def __init__(self, state: Any) -> None:
        from mtplx.server import openai as oa

        self.oa = oa
        self.state = state
        self.tokenizer = state.runtime.tokenizer
        self.sessions = state.sessions
        self.family = oa._model_family_for_state(state)
        self.reasoning_history_mode = oa._reasoning_history_mode(state)
        self.template_report: dict[str, Any] = {}

    @classmethod
    def for_pack(cls, pack: Path) -> "MtplxPath":
        from mtplx.artifacts import inspect_model
        from mtplx.backends.descriptors import descriptor_from_inspection, reasoning_policy_for_model
        from mtplx.runtime import _load_tokenizer_resilient
        from mtplx.server import openai as oa

        config = json.loads((pack / "config.json").read_text(encoding="utf-8"))
        tokenizer = _load_tokenizer_resilient(pack, config)
        inspection = inspect_model(pack).to_dict()
        backend_id = str(inspection.get("recommended_backend") or "qwen3_next")
        descriptor = descriptor_from_inspection(inspection)
        reasoning = reasoning_policy_for_model(inspection=inspection, descriptor=descriptor)
        args = oa.parse_args(["--model", str(pack), "--warmup-tokens", "0", "--backend-id", backend_id])
        # The launch defaults `mtplx serve` stamps for the family
        # (mtplx/commands/public.py:1321 _apply_backend_serve_defaults).
        args.reasoning_parser = reasoning.parser
        if getattr(args, "reasoning_effort", None) in (None, "auto") and reasoning.default_effort:
            args.reasoning_effort = reasoning.default_effort
        if descriptor.required_tool_prompt_mode is not None:
            args.tool_prompt_mode = descriptor.required_tool_prompt_mode
        state = SimpleNamespace(
            args=args,
            model_id=str(inspection.get("runtime_model") or pack.name),
            lock=Lock(),
            runtime=SimpleNamespace(model_path=pack, mtp_enabled=True, tokenizer=tokenizer, backend_id=backend_id),
            main_system_prompt_hash=None,
            has_foreground=lambda: False,
            sessions=_AuditSessions(),
        )
        # What ServerState does with the tokenizer at start-up (openai.py:3375-3400).
        template_report = oa._apply_chat_template_profile(tokenizer, args)
        state.template_hash = oa._template_hash(tokenizer)
        state.reasoning_history_scoped_capable = oa._template_supports_scoped_reasoning(tokenizer)
        path = cls(state)
        path.pack = pack
        path.template_report = template_report
        return path

    def prompt(self, messages: Sequence[dict[str, Any]], tools: list[dict[str, Any]] | None, *, lane: str,
               thinking: bool, reasoning_effort: str | None = None, use_session: bool = False,
               headers: dict[str, str] | None = None) -> dict[str, Any]:
        """The prompt ids for one request, built by the calls ``chat_completions``
        makes and in its order (mtplx/server/openai.py: resolve_request_policy at
        :31480, _vision_extract_and_flatten :31589, _encode_messages :31610,
        _maybe_canonicalize_committed_reasoning :31703)."""
        oa = self.oa
        state = self.state
        state.args.enable_thinking = bool(thinking)
        headers = dict(LANES[lane]["headers"]) if headers is None else dict(headers)
        body: dict[str, Any] = {"model": state.model_id, "messages": list(messages), "stream": True}
        if tools:
            body["tools"] = tools
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        request = oa.ChatCompletionRequest(**json.loads(json.dumps(body)))
        metadata = oa._request_metadata(request)
        policy = oa.resolve_request_policy(state, request, headers=headers, metadata=metadata, endpoint="chat")
        messages_for_generation, vision_images = oa._vision_extract_and_flatten(policy.messages_for_generation)
        assert not vision_images
        observability: dict[str, Any] = {}
        prompt_ids = oa._encode_messages(
            self.tokenizer,
            messages_for_generation,
            enable_thinking=policy.thinking_enabled,
            reasoning_effort=policy.reasoning_effort,
            strip_assistant_reasoning_history=state.args.strip_assistant_reasoning_history,
            scoped_reasoning_history=oa._reasoning_history_scoped_active(state),
            preserve_reasoning_history=oa._reasoning_history_preserve_echo_active(state),
            tools=policy.prompt_tool_specs,
            tool_choice=request.tool_choice,
            tool_prompt_mode=policy.template_tool_prompt_mode,
            template_observability=observability,
        )
        raw_ids = list(prompt_ids)
        canonicalized = None
        if use_session and not policy.background:
            session_id, _source = self.sessions.resolve_session_id(headers=headers)
            opencode_client = policy.opencode_client
            hermes_strip = bool(request.stream and policy.tools_active
                                and not policy.read_only_force_answer_contract_active
                                and oa._is_hermes_client(headers=headers, metadata=metadata))
            canonicalized = oa._maybe_canonicalize_committed_reasoning(
                state,
                messages=messages_for_generation,
                prompt_ids=prompt_ids,
                headers=headers,
                metadata=metadata,
                request=request,
                thinking_enabled=policy.thinking_enabled,
                reasoning_effort=policy.reasoning_effort,
                tools=policy.prompt_tool_specs,
                tool_choice=request.tool_choice,
                tool_prompt_mode=policy.template_tool_prompt_mode,
                template_observability=observability,
                transcript_stats=policy.transcript_stats,
                strip_tool_call_preamble_text=opencode_client or hermes_strip,
                session_id=session_id,
            )
            if canonicalized is not None:
                messages_for_generation, prompt_ids = canonicalized
        return {
            "ids": [int(t) for t in prompt_ids],
            "raw_ids": [int(t) for t in raw_ids],
            "thinking": bool(policy.thinking_enabled),
            "reasoning_effort": policy.reasoning_effort,
            "tool_prompt_mode": policy.template_tool_prompt_mode,
            "canonicalization": observability.get("committed_reasoning_canonicalization"),
            "observability": {k: v for k, v in observability.items() if k != "committed_reasoning_canonicalization"},
            "policy": policy,
            "messages_for_generation": list(messages_for_generation),
        }

    def postcommit_prediction(self, served: dict[str, Any], assistant_position: int) -> list[int] | None:
        """The prefix the postcommit banks once the assistant turn at
        ``assistant_position`` (in the served request's canonical messages) has
        been generated: what it predicts the NEXT request will start with
        (mtplx/server/openai.py _postcommit_next_turn_prefix_ids)."""
        oa = self.oa
        state = self.state
        policy = served["policy"]
        history = served["messages_for_generation"][: assistant_position + 1]
        if not history or history[-1].role != "assistant":
            return None
        return oa._postcommit_next_turn_prefix_ids(
            self.tokenizer,
            history,
            enable_thinking=policy.thinking_enabled,
            reasoning_effort=policy.reasoning_effort,
            strip_assistant_reasoning_history=state.args.strip_assistant_reasoning_history,
            scoped_reasoning_history=oa._reasoning_history_scoped_active(state),
            preserve_reasoning_history=oa._reasoning_history_preserve_echo_active(state),
            tools=policy.postcommit_tool_specs,
            assistant_tool_calls=history[-1].tool_calls,
            tool_prompt_mode=policy.postcommit_tool_prompt_mode,
        )


# ---------------------------------------------------------------------------
# Difference attribution
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    mechanism: str | None  # None = unexplained
    block: int
    role: str
    detail: str
    ref_text: str = ""
    mtplx_text: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = {"mechanism": self.mechanism or "UNEXPLAINED", "block": self.block, "role": self.role, "detail": self.detail}
        if self.mechanism is None:
            out["ref_text"] = self.ref_text[:400]
            out["mtplx_text"] = self.mtplx_text[:400]
        return out


def _blocks(ids: Sequence[int], im_start_id: int) -> list[list[int]]:
    blocks: list[list[int]] = []
    current: list[int] = []
    for token in ids:
        if token == im_start_id and current:
            blocks.append(current)
            current = []
        current.append(int(token))
    if current:
        blocks.append(current)
    return blocks


def _role_of(text: str) -> str:
    if text.startswith(IM_START):
        return text[len(IM_START):].split("\n", 1)[0].strip() or "?"
    return "?"


def _tools_section(text: str) -> tuple[int, int] | None:
    start = text.find("<tools>")
    end = text.find("</tools>")
    if start < 0 or end < 0 or end < start:
        return None
    return start + len("<tools>"), end


def _same_json_lines(ref_section: str, mtplx_section: str) -> bool:
    ref_lines = [line for line in ref_section.split("\n") if line.strip()]
    mtplx_lines = [line for line in mtplx_section.split("\n") if line.strip()]
    if len(ref_lines) != len(mtplx_lines) or not ref_lines:
        return False
    try:
        return all(json.loads(a) == json.loads(b) for a, b in zip(ref_lines, mtplx_lines))
    except ValueError:
        return False


def _char_hunks(ref_text: str, mtplx_text: str) -> list[tuple[str, str, int, int]]:
    """(ref_sub, mtplx_sub, ref_start, mtplx_start) for every differing span."""
    prefix = 0
    limit = min(len(ref_text), len(mtplx_text))
    while prefix < limit and ref_text[prefix] == mtplx_text[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and ref_text[-1 - suffix] == mtplx_text[-1 - suffix]:
        suffix += 1
    ref_mid = ref_text[prefix: len(ref_text) - suffix]
    mtplx_mid = mtplx_text[prefix: len(mtplx_text) - suffix]
    if len(ref_mid) * len(mtplx_mid) > 4_000_000 or not ref_mid or not mtplx_mid:
        return [(ref_mid, mtplx_mid, prefix, prefix)]
    matcher = difflib.SequenceMatcher(a=ref_mid, b=mtplx_mid, autojunk=False)
    hunks = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            hunks.append((ref_mid[i1:i2], mtplx_mid[j1:j2], prefix + i1, prefix + j1))
    return hunks


def _token_spans(ref_ids: Sequence[int], mtplx_ids: Sequence[int]) -> list[tuple[int, int, int, int]]:
    matcher = difflib.SequenceMatcher(a=list(ref_ids), b=list(mtplx_ids), autojunk=False)
    return [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]


def attribute(ref: Reference, ref_ids: Sequence[int], mtplx_ids: Sequence[int], *, thinking: bool,
              lane: str, committed: Sequence[int] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    ref_blocks = _blocks(ref_ids, ref.im_start_id)
    mtplx_blocks = _blocks(mtplx_ids, ref.im_start_id)
    matcher = difflib.SequenceMatcher(a=[tuple(b) for b in ref_blocks], b=[tuple(b) for b in mtplx_blocks], autojunk=False)
    mtplx_offset = [0]
    for block in mtplx_blocks:
        mtplx_offset.append(mtplx_offset[-1] + len(block))
    user_open = f"{IM_START}user\n"
    user_close = f"{IM_END}\n"

    def plain_user(text: str) -> bool:
        return text.startswith(user_open) and "<tool_response>" not in text and text.endswith(user_close)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        i, j = i1, j1
        while i < i2 and j < j2:
            ref_group = [ref_blocks[i]]
            mtplx_text = ref.decode(mtplx_blocks[j])
            # Consecutive plain user turns the server joined with a blank line.
            merged_text = ref.decode(ref_blocks[i])
            while (i + len(ref_group) < i2 and plain_user(merged_text)
                   and plain_user(ref.decode(ref_blocks[i + len(ref_group)]))
                   and merged_text != mtplx_text):
                following = ref.decode(ref_blocks[i + len(ref_group)])
                merged_text = merged_text[: -len(user_close)] + "\n\n" + following[len(user_open):]
                ref_group.append(ref_blocks[i + len(ref_group)])
            if len(ref_group) > 1 and merged_text == mtplx_text:
                findings.append(Finding("consecutive_role_merge", j, "user",
                                        f"{len(ref_group)} consecutive user turns rendered as one"))
            else:
                ref_group = [ref_blocks[i]]
                if ref_group[0] != mtplx_blocks[j]:
                    findings.extend(_attribute_pair(
                        ref, list(ref_group[0]), list(mtplx_blocks[j]), block_index=j, thinking=thinking,
                        lane=lane, committed=committed, mtplx_start=mtplx_offset[j],
                    ))
            i += len(ref_group)
            j += 1
        if i < i2 or j < j2:
            findings.append(Finding(None, j, "?", f"{i2 - i} reference turn(s) and {j2 - j} MTPLX turn(s) left unpaired",
                                    ref.decode([t for b in ref_blocks[i:i2] for t in b]),
                                    ref.decode([t for b in mtplx_blocks[j:j2] for t in b])))
    return findings


def _attribute_pair(ref: Reference, ref_ids: list[int], mtplx_ids: list[int], *, block_index: int,
                    thinking: bool, lane: str, committed: Sequence[int] | None,
                    mtplx_start: int) -> list[Finding]:
    ref_text = ref.decode(ref_ids)
    mtplx_text = ref.decode(mtplx_ids)
    role = _role_of(mtplx_text or ref_text)
    out: list[Finding] = []

    def found(mechanism: str | None, detail: str, a: str = "", b: str = "") -> None:
        out.append(Finding(mechanism, block_index, role, detail, a, b))

    if ref_text == mtplx_text:
        # Same text, different token boundaries.
        seam_char = len(THINK_SEAM) if ref_text.startswith(THINK_SEAM) else None
        for i1, i2, j1, j2 in _token_spans(ref_ids, mtplx_ids):
            span_start_char = len(ref.decode(mtplx_ids[:j1]))
            span_end_char = len(ref.decode(mtplx_ids[:j2]))
            absolute = mtplx_start + j1
            if committed is not None and list(committed[absolute: absolute + (j2 - j1)]) == mtplx_ids[j1:j2]:
                found("committed_id_splice", f"{i2 - i1} reference token(s) -> {j2 - j1} committed token(s): "
                      f"{ref.decode(ref_ids[i1:i2])!r}")
            elif seam_char is not None and span_start_char <= seam_char <= span_end_char and thinking:
                found("generation_seam_segmentation",
                      f"ref {[ref.decode([t]) for t in ref_ids[i1:i2]]!r} vs mtplx {[ref.decode([t]) for t in mtplx_ids[j1:j2]]!r}")
            elif seam_char is not None and span_start_char <= seam_char <= span_end_char:
                # Thinking off: no generation ever started after '<think>\n'
                # (the prompt carries the whole closed scaffold), so a split
                # there is not the documented generation-time boundary.
                found(None, "thinking off: the closed empty think scaffold is split after '<think>\\n'",
                      repr([ref.decode([t]) for t in ref_ids[i1:i2]]), repr([ref.decode([t]) for t in mtplx_ids[j1:j2]]))
            else:
                found(None, "same text, different token boundaries away from a generation seam",
                      repr([ref.decode([t]) for t in ref_ids[i1:i2]]), repr([ref.decode([t]) for t in mtplx_ids[j1:j2]]))
        return out

    work_ref, work_mtplx = ref_text, mtplx_text
    if role == "system":
        ref_tools = _tools_section(work_ref)
        mtplx_tools = _tools_section(work_mtplx)
        if ref_tools and mtplx_tools:
            ref_section = work_ref[ref_tools[0]: ref_tools[1]]
            mtplx_section = work_mtplx[mtplx_tools[0]: mtplx_tools[1]]
            if ref_section != mtplx_section:
                if _same_json_lines(ref_section, mtplx_section):
                    found("tool_schema_key_sort", "tool JSON values equal, object key order differs")
                    work_ref = work_ref[: ref_tools[0]] + mtplx_section + work_ref[ref_tools[1]:]
                else:
                    found(None, "tool schema JSON differs in value", ref_section, mtplx_section)
                    work_ref = work_ref[: ref_tools[0]] + mtplx_section + work_ref[ref_tools[1]:]
        elif ref_tools and not mtplx_tools:
            head = work_ref.find("# Tools\n\n")
            tail = work_ref.find("</IMPORTANT>")
            if head >= 0 and tail >= 0:
                tail += len("</IMPORTANT>")
                removed = work_ref[head:tail]
                # The template joins the tools block to the client's system
                # text with a blank line; that separator goes with the block.
                after = work_ref[tail:]
                if after.startswith("\n\n"):
                    after = after[2:]
                rest = work_ref[:head] + after
                found("compact_mode_template_tools_omitted", f"template tools block ({len(removed)} chars) not rendered")
                work_ref = rest

    for ref_sub, mtplx_sub, ref_at, mtplx_at in _char_hunks(work_ref, work_mtplx):
        if not ref_sub and not mtplx_sub:
            continue
        if role == "system" and not ref_sub.strip() and "MTPLX tool contract:" in mtplx_sub:
            found("tool_contract_system_injection", f"{len(mtplx_sub)} chars appended to the system message")
            continue
        if role == "system" and "MTPLX tool contract:" in mtplx_sub and ref_sub.strip() == "":
            found("tool_contract_system_injection", f"{len(mtplx_sub)} chars")
            continue
        if role == "assistant" and not mtplx_sub and lane == "opencode":
            before = work_ref[:ref_at]
            after = work_ref[ref_at + len(ref_sub):]
            if before.rstrip().endswith("</think>") and after.lstrip().startswith("<tool_call>"):
                found("opencode_tool_call_preamble_strip", f"visible text removed: {ref_sub.strip()[:80]!r}")
                continue
        if role == "assistant" and not thinking and not mtplx_sub.strip():
            before = work_ref[:ref_at]
            if "<think>" in before and "</think>" not in before[before.rfind("<think>"):]:
                found("thinking_off_history_reasoning_dropped", f"{len(ref_sub)} chars of echoed reasoning not rendered")
                continue
        if committed is not None and not ref_sub.strip() and not mtplx_sub.strip():
            found("committed_whitespace_restore", f"ref {ref_sub!r} vs mtplx {mtplx_sub!r}")
            continue
        found(None, "text differs", ref_sub if len(ref_sub) < 400 else ref_sub[:200] + " ... " + ref_sub[-150:],
              mtplx_sub if len(mtplx_sub) < 400 else mtplx_sub[:200] + " ... " + mtplx_sub[-150:])

    if not out:
        # Text equal after the documented system-block rewrites, ids differ.
        for i1, i2, j1, j2 in _token_spans(ref_ids, mtplx_ids):
            found(None, "ids differ with no text difference left to explain",
                  repr([ref.decode([t]) for t in ref_ids[i1:i2]]), repr([ref.decode([t]) for t in mtplx_ids[j1:j2]]))
    return out


def compare(ref: Reference, ref_ids: Sequence[int], mtplx_ids: Sequence[int], *, thinking: bool, lane: str,
            committed: Sequence[int] | None = None, window: int = 8) -> dict[str, Any]:
    equal = list(ref_ids) == list(mtplx_ids)
    record: dict[str, Any] = {
        "equal": equal, "ref_tokens": len(ref_ids), "mtplx_tokens": len(mtplx_ids),
        "mtplx_ids_sha256": hashlib.sha256(",".join(str(int(t)) for t in mtplx_ids).encode()).hexdigest()[:16],
    }
    if equal:
        record.update({"first_diff_index": None, "classification": "identical", "mechanisms": [], "body_identical": True})
        return record
    first = next((i for i, (a, b) in enumerate(zip(ref_ids, mtplx_ids)) if a != b), min(len(ref_ids), len(mtplx_ids)))
    lo = max(0, first - 4)
    record["first_diff_index"] = first
    record["ref_window"] = ref.decode(ref_ids[lo: first + window])
    record["mtplx_window"] = ref.decode(mtplx_ids[lo: first + window])
    record["ref_window_tokens"] = [ref.decode([t]) for t in ref_ids[lo: first + window]]
    record["mtplx_window_tokens"] = [ref.decode([t]) for t in mtplx_ids[lo: first + window]]
    record["text_equal"] = ref.decode(ref_ids) == ref.decode(mtplx_ids)
    findings = attribute(ref, ref_ids, mtplx_ids, thinking=thinking, lane=lane, committed=committed)
    # Everything after the system turn (every user, assistant and tool turn
    # and the generation prompt) equal to the reference, id for id.
    record["body_identical"] = bool(findings) and all(f.role == "system" for f in findings)
    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    for finding in findings:
        key = finding.mechanism or "UNEXPLAINED"
        counts[key] = counts.get(key, 0) + 1
        examples.setdefault(key, f"turn block {finding.block} ({finding.role}): {finding.detail}")
    record["mechanisms"] = [
        {"mechanism": k, "count": v, "where": MECHANISMS.get(k, {}).get("where"), "example": examples[k]}
        for k, v in sorted(counts.items())
    ]
    unexplained: dict[tuple[str, str, str], dict[str, Any]] = {}
    for finding in findings:
        if finding.mechanism is None:
            key = (finding.detail, finding.ref_text[:400], finding.mtplx_text[:400])
            entry = unexplained.setdefault(key, {**finding.as_dict(), "count": 0})
            entry["count"] += 1
    if unexplained or not findings:
        record["classification"] = "UNEXPLAINED"
        record["unexplained"] = list(unexplained.values())[:12] or [{"detail": "ids differ but no difference was located"}]
    else:
        record["classification"] = "explained"
    return record


# ---------------------------------------------------------------------------
# The committed-id splice scenario
# ---------------------------------------------------------------------------
def _noncanonical(ref: Reference, ids: list[int], *, mode: str, spacing: int = 7) -> tuple[list[int], int]:
    """Re-express ``ids`` with the same text and different token boundaries.

    split  one token -> the two tokens its BPE merge joined (a seam the
           tokenizer would never leave)
    join   two neighbours -> the single vocabulary token with their
           concatenated text, which the tokenizer never produces because its
           pre-tokenizer separates them (the 2026-09-08 Hermes receipt:
           '"Nothing' emitted as one token)
    """
    out: list[int] = []
    changed = 0
    since = spacing
    i = 0
    while i < len(ids):
        token = ids[i]
        text = ref.id_to_token.get(token)
        since += 1
        if token in ref.control_ids or text is None or since < spacing:
            out.append(token)
        elif mode == "split" and text in ref.merge_parts and len(text) > 2:
            left, right = ref.merge_parts[text]
            out.extend([ref.vocab[left], ref.vocab[right]])
            changed += 1
            since = 0
        elif (mode == "join" and i + 1 < len(ids) and ids[i + 1] not in ref.control_ids
              and ref.id_to_token.get(ids[i + 1]) is not None
              and (text + ref.id_to_token[ids[i + 1]]) in ref.vocab):
            out.append(ref.vocab[text + ref.id_to_token[ids[i + 1]]])
            changed += 1
            since = 0
            i += 1
        else:
            out.append(token)
        i += 1
    return out, changed


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


SPLICE_MODES = ("canonical", "split", "join", "split+eos", "trailing_newline")


def splice_scenario(ref: Reference, mtplx: MtplxPath, session: dict[str, Any], *, lane: str, thinking_mode: str,
                    mode: str) -> list[dict[str, Any]]:
    """Walk the session the way a live one runs: turn k's prompt plus the ids
    the model 'generated' for the next assistant message become the committed
    stream the following prompt is built against.

    canonical         the generated ids are the tokenizer's own encoding
    split / join      same text, token seams the tokenizer never produces
    split+eos         as split, and the committed stream keeps <|im_end|>
    trailing_newline  the model emitted a newline before ending a plain
                      answer; the response (and so the echoed history) is
                      stripped
    """
    thinking, keep_reasoning = THINKING_MODES[thinking_mode]
    messages = list(session["messages"]) if keep_reasoning else without_reasoning(session["messages"])
    tools = session["tools"]
    session_id = LANES[lane]["headers"].get("x-mtplx-session-id")
    if not session_id:
        return []
    mtplx.sessions.committed.pop(session_id, None)
    rows: list[dict[str, Any]] = []
    for position, cut in enumerate(turn_boundaries(messages)):
        served = mtplx.prompt(messages[:cut], tools, lane=lane, thinking=thinking, use_session=True)
        ref_text = ref.render(messages[:cut], tools, enable_thinking=served["thinking"],
                              reasoning_effort=served["reasoning_effort"])
        ref_ids = ref.encode(ref_text)
        committed = mtplx.sessions.committed.get(session_id)
        record = compare(ref, ref_ids, served["ids"], thinking=thinking, lane=lane, committed=committed)
        record["raw_equals_served"] = served["raw_ids"] == served["ids"]
        if committed:
            record["committed_len"] = len(committed)
            record["served_common_prefix"] = _common_prefix(served["ids"], committed)
            record["raw_common_prefix"] = _common_prefix(served["raw_ids"], committed)
            record["extends_committed"] = record["served_common_prefix"] == len(committed)
        record["canonicalization"] = served["canonicalization"]
        record.update({"scenario": f"session:{mode}", "turn": position, "messages": cut})
        rows.append(record)
        if cut >= len(messages) or messages[cut]["role"] != "assistant":
            break
        # What the model generated for messages[cut], as text: the rendered
        # conversation through that turn minus the prompt it continued.
        through = ref.render(messages[: cut + 1], tools, enable_thinking=served["thinking"],
                             reasoning_effort=served["reasoning_effort"], add_generation_prompt=False)
        if not through.startswith(ref_text):
            record["generated_text_note"] = "reference render of the turn does not extend the prompt text"
            mtplx.sessions.committed.pop(session_id, None)
            continue
        generated_text = through[len(ref_text):]
        if generated_text.endswith("\n"):
            generated_text = generated_text[:-1]  # the template's own newline after <|im_end|>
        body = generated_text[: -len(IM_END)] if generated_text.endswith(IM_END) else generated_text
        if mode == "trailing_newline" and not messages[cut].get("tool_calls"):
            body += "\n"
        generated_ids = ref.encode(body)
        changed = 0
        if mode in {"split", "join", "split+eos"}:
            generated_ids, changed = _noncanonical(ref, generated_ids, mode="join" if mode == "join" else "split")
        assert ref.decode(generated_ids) == body
        if mode == "split+eos":
            generated_ids = generated_ids + ref.encode(IM_END)
        record["next_turn_noncanonical_spots"] = changed
        mtplx.sessions.committed[session_id] = tuple(served["ids"]) + tuple(generated_ids)
    mtplx.sessions.committed.pop(session_id, None)
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def audit_pack(pack: Path, emit, *, sessions: list[dict[str, Any]] | None = None,
               scenarios: Sequence[str] = ("stateless", "session")) -> dict[str, Any]:
    started = time.time()
    ref = Reference(pack)
    mtplx = MtplxPath.for_pack(pack)
    sessions = sessions if sessions is not None else all_sessions()
    info = {
        "pack": pack.name,
        "family": mtplx.family,
        "reasoning_history_mode": mtplx.reasoning_history_mode,
        "launch_reasoning_effort": mtplx.state.args.reasoning_effort,
        "launch_tool_prompt_mode": mtplx.state.args.tool_prompt_mode,
        "chat_template_profile": mtplx.template_report,
        "reference_pretokenizer_regex_source": ref.regex_source,
        "server_template_equals_pack_template": getattr(mtplx.tokenizer, "chat_template", None) == ref.template,
    }
    # The two tokenizers must agree on plain text before any prompt is compared.
    probe_texts = [m["content"] for s in sessions for m in s["messages"] if isinstance(m.get("content"), str)]
    probe_texts += [m["reasoning_content"] for s in sessions for m in s["messages"] if m.get("reasoning_content")]
    info["server_tokenizer_vs_reference_encoder_disagreements"] = sum(
        1 for text in probe_texts
        if [int(t) for t in mtplx.oa._encode_rendered_chat_text(mtplx.tokenizer, text)] != ref.encode(text)
    )
    info["probe_texts"] = len(probe_texts)
    # For information: what plain transformers (no MTPLX loader) would make of
    # the same texts. Its Qwen2Tokenizer class rebuilds the pre-tokenizer from
    # its own regex (mtplx/runtime.py:1324 restore_qwen3_pretokenizer).
    info["plain_transformers_encoder_disagreements"] = sum(
        1 for text in probe_texts if list(ref.hf.encode(text, add_special_tokens=False)) != ref.encode(text)
    )
    # The template's own default equals the explicit switch the server passes.
    sample = sessions[0]
    info["reference_preserve_thinking_default_equals_true"] = (
        ref.render(sample["messages"], sample["tools"], enable_thinking=True, reasoning_effort=None, preserve_thinking=None)
        == ref.render(sample["messages"], sample["tools"], enable_thinking=True, reasoning_effort=None, preserve_thinking=True)
    )
    emit({"kind": "pack", **info})

    summary: dict[tuple[str, str, str, str], dict[str, int]] = {}

    def tally(key: tuple[str, str, str, str], record: dict[str, Any]) -> None:
        bucket = summary.setdefault(key, {"turns": 0, "identical": 0, "explained": 0, "UNEXPLAINED": 0,
                                          "body_identical": 0})
        bucket["turns"] += 1
        bucket[record["classification"]] += 1
        bucket["body_identical"] += int(bool(record.get("body_identical")))
        for item in record.get("mechanisms", []):
            bucket[f"m:{item['mechanism']}"] = bucket.get(f"m:{item['mechanism']}", 0) + 1
        if record.get("next_turn_noncanonical_spots"):
            bucket["noncanonical_spots"] = bucket.get("noncanonical_spots", 0) + int(record["next_turn_noncanonical_spots"])
        if "postcommit_prefix_ok" in record:
            bucket["with_postcommit"] = bucket.get("with_postcommit", 0) + 1
            bucket["postcommit_ok"] = bucket.get("postcommit_ok", 0) + int(record["postcommit_prefix_ok"])
        if "extends_committed" in record:
            bucket["with_committed"] = bucket.get("with_committed", 0) + 1
            bucket["extends_committed"] = bucket.get("extends_committed", 0) + int(record["extends_committed"])

    if "stateless" in scenarios:
        for session in sessions:
            for lane in SHAPE_LANES[session["shape"]]:
                for thinking_mode, (thinking, keep_reasoning) in THINKING_MODES.items():
                    messages_all = session["messages"] if keep_reasoning else without_reasoning(session["messages"])
                    if thinking_mode == "off_after_on" and not any(m.get("reasoning_content") for m in session["messages"]):
                        continue
                    for position, cut in enumerate(turn_boundaries(messages_all)):
                        messages = messages_all[:cut]
                        served = mtplx.prompt(messages, session["tools"], lane=lane, thinking=thinking)
                        ref_text = ref.render(messages, session["tools"], enable_thinking=served["thinking"],
                                              reasoning_effort=served["reasoning_effort"])
                        record = compare(ref, ref.encode(ref_text), served["ids"], thinking=thinking, lane=lane)
                        record.update({
                            "kind": "turn", "pack": pack.name, "session": session["name"], "lane": lane,
                            "thinking": thinking_mode, "turn": position, "messages": cut, "scenario": "stateless",
                            "tool_prompt_mode": served["tool_prompt_mode"], "reasoning_effort": served["reasoning_effort"],
                        })
                        # Cache side of the same question: what the postcommit
                        # banked after the previous assistant turn must be a
                        # prefix of this request.
                        canonical = served["messages_for_generation"]
                        last_assistant = max((i for i, m in enumerate(canonical) if m.role == "assistant"), default=None)
                        if last_assistant is not None:
                            predicted = mtplx.postcommit_prediction(served, last_assistant)
                            if predicted:
                                record["postcommit_prefix_tokens"] = len(predicted)
                                record["postcommit_prefix_ok"] = served["ids"][: len(predicted)] == predicted
                        emit(record)
                        tally((session["name"], lane, thinking_mode, "stateless"), record)

    if "session" in scenarios:
        for session in sessions:
            lane = session["shape"] if session["shape"] in LANES else "pi"
            if len(turn_boundaries(session["messages"])) < 2:
                continue
            for thinking_mode in ("off",) if session["shape"] == "hermes" else ("on", "off"):
                for mode in SPLICE_MODES:
                    for record in splice_scenario(ref, mtplx, session, lane=lane, thinking_mode=thinking_mode, mode=mode):
                        record.update({"kind": "turn", "pack": pack.name, "session": session["name"], "lane": lane,
                                       "thinking": thinking_mode})
                        emit(record)
                        tally((session["name"], lane, thinking_mode, record["scenario"]), record)

    info["seconds"] = round(time.time() - started, 1)
    return {"info": info, "summary": summary}


def _print_summary(results: list[dict[str, Any]], stream) -> None:
    for result in results:
        info = result["info"]
        print(f"\n== {info['pack']}  family={info['family']}  history={info['reasoning_history_mode']}  "
              f"effort={info['launch_reasoning_effort']}  ({info['seconds']} s)", file=stream)
        print(f"   server tokenizer vs reference encoder: {info['server_tokenizer_vs_reference_encoder_disagreements']} "
              f"disagreements on {info['probe_texts']} texts (plain transformers: "
              f"{info['plain_transformers_encoder_disagreements']}); reference regex from "
              f"{info['reference_pretokenizer_regex_source']}", file=stream)
        header = (f"   {'session':<21}{'lane':<10}{'think':<13}{'scenario':<25}{'turns':>5}{'ident':>6}{'expl':>5}"
                  f"{'UNEXPL':>7}{'body=':>6}{'cache':>7}  mechanisms (turns)")
        print(header, file=stream)
        for (session, lane, thinking, scenario), bucket in result["summary"].items():
            mechanisms = ", ".join(f"{k[2:]} {v}" for k, v in sorted(bucket.items()) if k.startswith("m:"))
            if bucket.get("noncanonical_spots"):
                mechanisms = f"[{bucket['noncanonical_spots']} non-canonical spots] " + mechanisms
            extends = f"{bucket['extends_committed']}/{bucket['with_committed']}" if bucket.get("with_committed") else "-"
            if bucket.get("with_postcommit"):
                extends = f"{bucket['postcommit_ok']}/{bucket['with_postcommit']}"
            print(f"   {session:<21}{lane:<10}{thinking:<13}{scenario:<25}{bucket['turns']:>5}{bucket['identical']:>6}"
                  f"{bucket['explained']:>5}{bucket['UNEXPLAINED']:>7}{bucket['body_identical']:>6}{extends:>7}  {mechanisms}",
                  file=stream)


def _hermetic_environment() -> None:
    """Run as a script only (never at import: a test process importing this
    module must keep its own environment). Synthetic requests never enter a
    real request log or flight recorder, and every encode is a fresh one."""
    os.environ["MTPLX_REQUEST_LOG_JSONL"] = "off"
    os.environ["MTPLX_FLIGHT_RECORDER"] = "off"
    os.environ["MTPLX_CHAT_ENCODE_CACHE"] = "off"


def main(argv: list[str] | None = None) -> int:
    _hermetic_environment()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", action="append", default=[], help="Pack directory (repeatable). Default: the two shipped packs under ~/.mtplx/models that exist.")
    parser.add_argument("--out", default="-", help="JSONL output path ('-' = stdout)")
    parser.add_argument("--summary-json", default=None, help="Write the summary table as JSON")
    parser.add_argument("--only-session", action="append", default=[], help="Limit to the named synthetic session(s)")
    parser.add_argument("--scenario", action="append", default=[], choices=["stateless", "session"],
                        help="stateless = every turn boundary rendered on its own; session = the same walk with a "
                        "committed stream between turns (committed-think substitution and the committed-id splice). "
                        "Default: both.")
    args = parser.parse_args(argv)

    packs = [Path(p).expanduser() for p in args.pack]
    if not packs:
        root = Path.home() / ".mtplx" / "models"
        packs = [root / name for name in DEFAULT_PACKS if (root / name / "tokenizer.json").exists()]
    if not packs:
        print("no pack found; pass --pack", file=sys.stderr)
        return 2
    sessions = all_sessions()
    if args.only_session:
        sessions = [s for s in sessions if s["name"] in set(args.only_session)]

    sink = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    results = []
    try:
        def emit(record: dict[str, Any]) -> None:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        for pack in packs:
            results.append(audit_pack(pack, emit, sessions=sessions, scenarios=tuple(args.scenario) or ("stateless", "session")))
    finally:
        if sink is not sys.stdout:
            sink.close()
    _print_summary(results, sys.stderr)
    if args.summary_json:
        payload = [{"info": r["info"], "summary": [{"session": k[0], "lane": k[1], "thinking": k[2], "scenario": k[3], **v} for k, v in r["summary"].items()]} for r in results]
        Path(args.summary_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    unexplained = sum(bucket["UNEXPLAINED"] for r in results for bucket in r["summary"].values())
    return 1 if unexplained else 0


if __name__ == "__main__":
    raise SystemExit(main())
