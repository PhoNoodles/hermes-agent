"""Deterministic parent-boundary tests for exact-child delegation lifecycle."""

import json
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    _enforce_foreground_result_identity,
    _run_single_child,
    delegate_task,
)
from tools.process_registry import process_registry


def _parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._interrupt_requested = False
    parent._memory_manager = None
    parent._print_fn = None
    parent._delegate_spinner = None
    parent.tool_progress_callback = None
    parent.session_id = "parent-session"
    parent.session_estimated_cost_usd = 0.0
    parent.session_cost_source = "none"
    parent.session_cost_status = "unknown"
    return parent


def _child(subagent_id, run):
    child = MagicMock()
    child._subagent_id = subagent_id
    child._delegate_depth = 1
    child._delegate_role = "leaf"
    child._credential_pool = None
    child._delegate_saved_tool_names = []
    child.session_id = f"session-{subagent_id}"
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    child.session_estimated_cost_usd = 0.0
    child.session_cost_status = "unknown"
    child.model = "test-model"
    child.tool_progress_callback = None
    child.get_activity_summary.return_value = {
        "current_tool": None,
        "api_call_count": 1,
        "max_iterations": 1,
        "last_activity_desc": "",
    }
    child.run_conversation.side_effect = run
    return child


class TestExactChildLifecycle:
    def test_agent_dispatch_stays_pending_until_exact_child_releases(self):
        from run_agent import AIAgent

        release = threading.Event()
        started = threading.Event()
        stale = {"type": "async_delegation", "delegation_id": "stale-B"}
        process_registry.completion_queue.put(stale)

        def run(**kwargs):
            assert kwargs["task_id"] == "exact-A"
            started.set()
            assert release.wait(2)
            return {"final_response": "A terminal", "completed": True, "api_calls": 1}

        child = _child("exact-A", run)
        parent = _parent()
        try:
            with (
                patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=child),
                patch(
                    "tools.delegate_tool._resolve_delegation_credentials",
                    return_value={
                        "model": None,
                        "provider": None,
                        "base_url": None,
                        "api_key": None,
                        "api_mode": None,
                        "request_overrides": None,
                        "max_output_tokens": None,
                        "command": None,
                        "args": None,
                    },
                ),
                patch("tools.async_delegation.dispatch_async_delegation_batch") as dispatch,
                ThreadPoolExecutor(max_workers=1) as executor,
            ):
                # Even a model-emitted background request enters the
                # authoritative foreground path: _dispatch_delegate_task calls
                # the real delegate_task with background=False and waits for
                # this invocation's exact child.
                future = executor.submit(
                    AIAgent._dispatch_delegate_task,
                    parent,
                    {"goal": "A", "background": True},
                )
                assert started.wait(2)
                assert not future.done()
                assert process_registry.completion_queue.queue[0] is stale
                release.set()
                payload = json.loads(future.result(timeout=2))

            dispatch.assert_not_called()
            assert payload["results"][0]["subagent_id"] == "exact-A"
            assert payload["results"][0]["summary"] == "A terminal"
        finally:
            assert process_registry.completion_queue.get_nowait() is stale

    def test_ab_b_first_demo(self):
        release_a = threading.Event()
        started_a = threading.Event()
        ordering = []

        def run_a(**kwargs):
            assert kwargs["task_id"] == "exact-A"
            started_a.set()
            assert release_a.wait(2)
            ordering.append("A")
            return {"final_response": "A terminal", "completed": True, "api_calls": 1}

        def run_b(**kwargs):
            assert kwargs["task_id"] == "exact-B"
            ordering.append("B")
            return {"final_response": "B terminal", "completed": True, "api_calls": 1}

        parent = _parent()
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(_run_single_child, 0, "A", _child("exact-A", run_a), parent)
            assert started_a.wait(2)
            result_b = executor.submit(
                _run_single_child, 1, "B", _child("exact-B", run_b), parent
            ).result(timeout=2)
            print(f"B completed first: {result_b['subagent_id']}; ordering={ordering}")
            assert not future_a.done()
            release_a.set()
            result_a = future_a.result(timeout=2)
            print(f"A released second: {result_a['subagent_id']}; ordering={ordering}")

        assert ordering == ["B", "A"]
        assert (result_a["subagent_id"], result_b["subagent_id"]) == ("exact-A", "exact-B")
        assert (result_a["summary"], result_b["summary"]) == ("A terminal", "B terminal")

    def test_foreground_mismatch_is_rejected_locally(self):
        stale = {"type": "async_delegation", "delegation_id": "unrelated"}
        process_registry.completion_queue.put(stale)
        entries = [{
            "task_index": 0,
            "subagent_id": "exact-B",
            "status": "completed",
            "summary": "wrong child",
        }]
        try:
            child = _child("exact-A", lambda **_: None)
            _enforce_foreground_result_identity([(0, {"goal": "A"}, child)], entries)
            assert entries[0]["subagent_id"] == "exact-A"
            assert entries[0]["status"] == "error"
            assert entries[0]["exit_reason"] == "correlation_mismatch"
            assert process_registry.completion_queue.queue[0] is stale
        finally:
            assert process_registry.completion_queue.get_nowait() is stale

    def test_failure_and_timeout_retain_exact_ids(self):
        parent = _parent()
        failed = _child(
            "exact-failure",
            lambda **_: (_ for _ in ()).throw(RuntimeError("child failed")),
        )
        failure = _run_single_child(0, "failure", failed, parent)

        release = threading.Event()
        timed_out = _child("exact-timeout", lambda **_: release.wait(2))
        with patch("tools.delegate_tool._get_child_timeout", return_value=0.01):
            timeout = _run_single_child(1, "timeout", timed_out, parent)
        release.set()

        assert failure["subagent_id"] == "exact-failure"
        assert failure["status"] == "error"
        assert "child failed" in failure["error"]
        assert timeout["subagent_id"] == "exact-timeout"
        assert timeout["status"] == "timeout"
        assert timeout["timeout_seconds"] == 0.01

    def test_explicit_direct_background_true_remains_async(self):
        parent = _parent()
        child = _child(
            "exact-background",
            lambda **_: (_ for _ in ()).throw(AssertionError("ran synchronously")),
        )
        captured = {}

        def dispatch(**kwargs):
            captured.update(kwargs)
            return {"status": "dispatched", "delegation_id": "deleg-exact"}

        with (
            patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=child),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value={
                    "model": None,
                    "provider": None,
                    "base_url": None,
                    "api_key": None,
                    "api_mode": None,
                    "request_overrides": None,
                    "max_output_tokens": None,
                    "command": None,
                    "args": None,
                },
            ),
            patch("tools.async_delegation.dispatch_async_delegation_batch", side_effect=dispatch),
        ):
            payload = json.loads(
                delegate_task(goal="background", background=True, parent_agent=parent)
            )

        assert payload["status"] == "dispatched"
        assert payload["mode"] == "background"
        assert payload["subagent_ids"] == ["exact-background"]
        assert captured["runner"] is not None
        child.run_conversation.assert_not_called()
