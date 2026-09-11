"""engy-miner: what a generation that ran out of budget reports.

A reasoning model given a small `max_tokens` spends the whole budget inside the
think block and emits no content at all. Reported as `finish_reason: "stop"` —
which is what this lane did — a client that trusts the field reads a cut-off
turn as a completed one, and a client that renders `content` shows the model's
raw chain of thought as the answer. Both are pinned here.

The functions under test are pure string handling, but the module they live in
imports torch, transformers and toploc at the top for the proof path, so the
import is stubbed rather than made a precondition of running the suite.
"""
import sys
import types

import pytest


@pytest.fixture(scope="module")
def em():
    heavy = {}
    for name in ("torch", "requests", "websockets"):
        heavy[name] = types.ModuleType(name)
    heavy["toploc"] = types.ModuleType("toploc")
    heavy["toploc"].build_proofs_base64 = lambda *a, **k: []
    heavy["transformers"] = types.ModuleType("transformers")
    heavy["transformers"].AutoTokenizer = object
    saved = {k: sys.modules.get(k) for k in heavy}
    sys.modules.update({k: v for k, v in heavy.items() if saved[k] is None})
    try:
        from miner import engy_miner
        yield engy_miner
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)


def test_budget_spent_inside_think_reports_length(em):
    """The reported case: cap 16, the model never reaches `</think>`."""
    msg, finish = em._assemble_message(
        "Okay, the user is asking about the capital of",
        None, think_open=True, truncated=True)
    assert finish == "length"
    # not the chain of thought dressed up as an answer
    assert msg["content"] is None
    assert msg["reasoning_content"] == "Okay, the user is asking about the capital of"


def test_budget_spent_after_think_closed_reports_length(em):
    msg, finish = em._assemble_message(
        "Let me compute.</think>The answer is 3",
        None, think_open=True, truncated=True)
    assert finish == "length"
    assert msg["content"] == "The answer is 3"
    assert msg["reasoning_content"] == "Let me compute."


def test_model_stopped_on_its_own_reports_stop(em):
    """Enough budget to finish: unchanged behaviour."""
    msg, finish = em._assemble_message(
        "Let me compute.</think>The answer is 391.",
        None, think_open=True, truncated=False)
    assert finish == "stop"
    assert msg["content"] == "The answer is 391."


def test_truncation_outranks_tool_calls(em):
    """sglang's OpenAI route promotes to "tool_calls" only from "stop"; a call
    parsed out of a truncated generation is still a truncated generation."""
    tools = [{"type": "function",
              "function": {"name": "get_weather",
                           "parameters": {"properties": {"city": {"type": "string"}}}}}]
    text = ('</think>Checking.<tool_call><function=get_weather>'
            '<parameter=city>Paris</parameter></function></tool_call>')
    msg, finish = em._assemble_message(text, tools, think_open=True, truncated=False)
    assert finish == "tool_calls"
    assert msg["tool_calls"]
    _, finish = em._assemble_message(text, tools, think_open=True, truncated=True)
    assert finish == "length"
