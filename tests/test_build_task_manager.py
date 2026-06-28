from __future__ import annotations

import asyncio
import time

import pytest

from treefyit.server.build_tasks import (
    BuildTask,
    BuildTaskExecutionError,
    BuildTaskManager,
    validate_tasks,
)


def test_validate_tasks_rejects_missing_dependency():
    with pytest.raises(ValueError, match="unknown dependencies"):
        validate_tasks(
            [BuildTask("build", lambda _context: None, depends_on=("parse",))]
        )


def test_validate_tasks_rejects_dependency_cycle():
    with pytest.raises(ValueError, match="cycle"):
        validate_tasks(
            [
                BuildTask("a", lambda _context: None, depends_on=("b",)),
                BuildTask("b", lambda _context: None, depends_on=("a",)),
            ]
        )


def test_build_task_manager_runs_dependencies_in_order():
    events: list[str] = []

    async def run():
        manager = BuildTaskManager()
        context = await manager.run(
            [
                BuildTask("parse", lambda _context: "sections"),
                BuildTask(
                    "build",
                    lambda context: f"tree from {context['parse']}",
                    depends_on=("parse",),
                ),
            ],
            on_event=lambda event: events.append(f"{event['type']}:{event['task']}"),
        )
        return context

    context = asyncio.run(run())

    assert context == {"parse": "sections", "build": "tree from sections"}
    assert events == [
        "task_start:parse",
        "task_done:parse",
        "task_start:build",
        "task_done:build",
    ]


def test_build_task_manager_runs_independent_tasks_in_parallel():
    def slow_task(_context):
        time.sleep(0.12)
        return "ok"

    async def run():
        started_at = time.perf_counter()
        context = await BuildTaskManager(max_concurrency=2).run(
            [
                BuildTask("a", slow_task),
                BuildTask("b", slow_task),
            ]
        )
        return context, time.perf_counter() - started_at

    context, elapsed_sec = asyncio.run(run())

    assert context == {"a": "ok", "b": "ok"}
    assert elapsed_sec < 0.22


def test_build_task_manager_stops_on_task_error():
    def fail_build(_context):
        raise RuntimeError("boom")

    async def run():
        return await BuildTaskManager().run(
            [
                BuildTask("parse", lambda _context: "sections"),
                BuildTask("build", fail_build, depends_on=("parse",)),
            ]
        )

    with pytest.raises(BuildTaskExecutionError, match="boom"):
        asyncio.run(run())
