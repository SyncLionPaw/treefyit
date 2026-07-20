"""Task orchestration for API build workflows."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

BuildContext = dict[str, Any]
BuildTaskEventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass(frozen=True)
class BuildTask:
    name: str
    run: Callable[[Mapping[str, Any]], Any]
    depends_on: tuple[str, ...] = ()
    description: str = ""


@dataclass
class BuildTaskRecord:
    name: str
    depends_on: tuple[str, ...]
    status: str
    description: str = ""
    elapsed_ms: float | None = None
    error: str | None = None

    def to_event(self, event_type: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": event_type,
            "stage": self.name,
            "task": self.name,
            "depends_on": list(self.depends_on),
            "status": self.status,
        }
        if self.description:
            payload["description"] = self.description
        if self.elapsed_ms is not None:
            payload["elapsed_ms"] = round(self.elapsed_ms, 2)
        if self.error:
            payload["message"] = self.error
        return payload


class BuildTaskExecutionError(RuntimeError):
    def __init__(self, task_name: str, message: str, records: list[BuildTaskRecord]):
        super().__init__(message)
        self.task_name = task_name
        self.records = records


class BuildTaskManager:
    def __init__(
        self,
        *,
        max_concurrency: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.max_concurrency = max_concurrency or 4
        self.logger = logger or logging.getLogger("treefyit.server.build_tasks")

    async def run(
        self,
        tasks: list[BuildTask],
        *,
        on_event: BuildTaskEventHandler | None = None,
    ) -> BuildContext:
        task_map = validate_tasks(tasks)
        context: BuildContext = {}
        pending = dict(task_map)
        running: dict[str, asyncio.Task[tuple[str, Any, BuildTaskRecord]]] = {}
        records: list[BuildTaskRecord] = []

        while pending or running:
            ready = [
                task
                for task in pending.values()
                if all(dep in context for dep in task.depends_on)
            ]
            slots = self.max_concurrency - len(running)
            for task in ready[:slots]:
                pending.pop(task.name)
                running[task.name] = asyncio.create_task(
                    self.run_one(task, context, on_event)
                )

            if not running:
                cycle = ", ".join(sorted(pending))
                raise ValueError(f"build task dependency cycle detected: {cycle}")

            done, _pending = await asyncio.wait(
                running.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for completed_task in done:
                task_name = next(
                    name for name, task in running.items() if task is completed_task
                )
                running.pop(task_name)
                try:
                    name, value, record = completed_task.result()
                except Exception as exc:  # noqa: BLE001
                    for task in running.values():
                        task.cancel()
                    if running:
                        await asyncio.gather(*running.values(), return_exceptions=True)
                    if isinstance(exc, BuildTaskExecutionError):
                        raise exc
                    raise BuildTaskExecutionError(task_name, str(exc), records) from exc
                records.append(record)
                context[name] = value

        return context

    async def run_one(
        self,
        task: BuildTask,
        context: Mapping[str, Any],
        on_event: BuildTaskEventHandler | None,
    ) -> tuple[str, Any, BuildTaskRecord]:
        start_record = BuildTaskRecord(
            name=task.name,
            depends_on=task.depends_on,
            status="running",
            description=task.description,
        )
        await emit_task_event(on_event, start_record.to_event("task_start"))
        self.logger.info(
            "build task started task=%s depends_on=%s",
            task.name,
            ",".join(task.depends_on) or "-",
        )

        started_at = time.perf_counter()
        try:
            value = await run_task_callable(task.run, context)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            error_record = BuildTaskRecord(
                name=task.name,
                depends_on=task.depends_on,
                status="error",
                description=task.description,
                elapsed_ms=elapsed_ms,
                error=str(exc),
            )
            await emit_task_event(on_event, error_record.to_event("task_error"))
            self.logger.exception(
                "build task failed task=%s elapsed_ms=%.2f",
                task.name,
                elapsed_ms,
            )
            raise BuildTaskExecutionError(task.name, str(exc), [error_record]) from exc

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        done_record = BuildTaskRecord(
            name=task.name,
            depends_on=task.depends_on,
            status="done",
            description=task.description,
            elapsed_ms=elapsed_ms,
        )
        await emit_task_event(on_event, done_record.to_event("task_done"))
        self.logger.info(
            "build task completed task=%s elapsed_ms=%.2f",
            task.name,
            elapsed_ms,
        )
        return task.name, value, done_record


async def run_task_callable(
    task_callable: Callable[[Mapping[str, Any]], Any],
    context: Mapping[str, Any],
) -> Any:
    if not inspect.iscoroutinefunction(task_callable):
        return await asyncio.to_thread(task_callable, context)

    result = task_callable(context)
    if inspect.isawaitable(result):
        return await result
    return result


async def emit_task_event(
    on_event: BuildTaskEventHandler | None,
    payload: dict[str, Any],
) -> None:
    if on_event is None:
        return
    result = on_event(payload)
    if inspect.isawaitable(result):
        await result


def validate_tasks(tasks: list[BuildTask]) -> dict[str, BuildTask]:
    task_map: dict[str, BuildTask] = {}
    for task in tasks:
        if not task.name:
            raise ValueError("build task name is required")
        if task.name in task_map:
            raise ValueError(f"duplicate build task: {task.name}")
        task_map[task.name] = task

    for task in tasks:
        missing = [dep for dep in task.depends_on if dep not in task_map]
        if missing:
            raise ValueError(
                f"build task {task.name} has unknown dependencies: {', '.join(missing)}"
            )

    detect_cycles(task_map)
    return task_map


def detect_cycles(task_map: Mapping[str, BuildTask]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, path: list[str]) -> None:
        if name in visited:
            return
        if name in visiting:
            cycle = " -> ".join([*path, name])
            raise ValueError(f"build task dependency cycle detected: {cycle}")
        visiting.add(name)
        for dep in task_map[name].depends_on:
            visit(dep, [*path, name])
        visiting.remove(name)
        visited.add(name)

    for name in task_map:
        visit(name, [])


__all__ = [
    "BuildContext",
    "BuildTask",
    "BuildTaskExecutionError",
    "BuildTaskManager",
    "BuildTaskRecord",
    "validate_tasks",
]
