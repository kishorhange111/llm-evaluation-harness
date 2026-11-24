"""Unified registry and plugin discovery for tasks, models, metrics, and runner hooks.

Provides a lightweight, typed ``Registry`` with metadata tagging, duplicate
registration guards, decorator registration, and ``importlib.metadata`` entry-point
discovery so third-party packages can extend ``evalkit`` without modifying core code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")

PLUGIN_ENTRY_POINT_GROUP = "evalkit.plugins"


@dataclass(frozen=True)
class RegistryEntry(Generic[T]):
    """A registered component together with optional discovery metadata."""

    name: str
    target: T
    category: str = "general"
    tags: tuple[str, ...] = ()
    description: str = ""
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)


class Registry(Generic[T]):
    """Named component registry with decorator support, filtering, and validation."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, RegistryEntry[T]] = {}

    def register(
        self,
        name: str,
        target: T | None = None,
        *,
        category: str = "general",
        tags: tuple[str, ...] | list[str] = (),
        description: str = "",
        version: str = "1.0",
        override: bool = False,
        **extra: Any,
    ) -> Any:
        """Register ``target`` directly or return a decorator when ``target`` is omitted."""

        def _attach(obj: T) -> T:
            if not name or not name.strip():
                raise ValueError(f"{self.kind} registry name must be non-empty")
            if name in self._entries and not override:
                raise ValueError(f"{self.kind} {name!r} is already registered (pass override=True to replace)")
            doc = description
            if not doc and getattr(obj, "__doc__", None):
                doc = str(obj.__doc__).strip().splitlines()[0]
            self._entries[name] = RegistryEntry(
                name=name,
                target=obj,
                category=category,
                tags=tuple(tags),
                description=doc,
                version=version,
                metadata=dict(extra),
            )
            return obj

        if target is not None:
            return _attach(target)
        return _attach

    def unregister(self, name: str) -> RegistryEntry[T]:
        """Remove and return a registered entry by name."""
        try:
            return self._entries.pop(name)
        except KeyError:
            raise ValueError(f"unknown {self.kind} {name!r}; available: {', '.join(sorted(self._entries))}") from None

    def get(self, name: str) -> T:
        """Return the registered target for ``name`` or raise ``ValueError``."""
        try:
            return self._entries[name].target
        except KeyError:
            raise ValueError(f"unknown {self.kind} {name!r}; available: {', '.join(sorted(self._entries))}") from None

    def get_entry(self, name: str) -> RegistryEntry[T]:
        """Return the full ``RegistryEntry`` (including metadata) for ``name``."""
        try:
            return self._entries[name]
        except KeyError:
            raise ValueError(f"unknown {self.kind} {name!r}; available: {', '.join(sorted(self._entries))}") from None

    def names(self, *, category: str | None = None, tag: str | None = None) -> list[str]:
        """List registered names in sorted order, optionally filtered by category or tag."""
        return [e.name for e in self.entries(category=category, tag=tag)]

    def entries(self, *, category: str | None = None, tag: str | None = None) -> list[RegistryEntry[T]]:
        """List registered entries in sorted order, optionally filtered by category or tag."""
        out = []
        for name in sorted(self._entries):
            entry = self._entries[name]
            if category is not None and entry.category != category:
                continue
            if tag is not None and tag not in entry.tags:
                continue
            out.append(entry)
        return out

    def as_dict(self) -> dict[str, T]:
        """Return a shallow mapping of ``name -> target``."""
        return {name: entry.target for name, entry in self._entries.items()}

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)


@runtime_checkable
class RunnerHook(Protocol):
    """Lifecycle hook protocol for extending ``evalkit.runner.run``."""

    def on_run_start(self, run_name: str, config: dict[str, Any]) -> None: ...

    def on_task_start(self, model_name: str, task_label: str, task_spec: dict[str, Any]) -> None: ...

    def on_task_end(self, model_name: str, task_label: str, metrics: dict[str, Any]) -> None: ...

    def on_run_end(self, summary: dict[str, Any], comparisons: dict[str, Any]) -> None: ...


class BaseRunnerHook:
    """No-op base class for hooks that only override a subset of lifecycle events."""

    def on_run_start(self, run_name: str, config: dict[str, Any]) -> None:
        pass

    def on_task_start(self, model_name: str, task_label: str, task_spec: dict[str, Any]) -> None:
        pass

    def on_task_end(self, model_name: str, task_label: str, metrics: dict[str, Any]) -> None:
        pass

    def on_run_end(self, summary: dict[str, Any], comparisons: dict[str, Any]) -> None:
        pass


TASK_REGISTRY: Registry[type[Any]] = Registry("task")
MODEL_REGISTRY: Registry[Callable[..., Any]] = Registry("backend")
METRIC_REGISTRY: Registry[Callable[..., Any]] = Registry("metric")


def register_task(
    name: str | None = None,
    *,
    category: str = "benchmark",
    tags: tuple[str, ...] | list[str] = (),
    description: str = "",
    override: bool = False,
) -> Callable[[type[Any]], type[Any]]:
    """Decorator to register a :class:`~evalkit.tasks.base.Task` subclass."""

    def _decorator(cls: type[Any]) -> type[Any]:
        task_name = name or getattr(cls, "name", cls.__name__.lower())
        TASK_REGISTRY.register(task_name, cls, category=category, tags=tags, description=description, override=override)
        return cls

    return _decorator


def register_model(
    name: str,
    *,
    category: str = "model",
    tags: tuple[str, ...] | list[str] = (),
    description: str = "",
    override: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a model backend factory."""
    return MODEL_REGISTRY.register(name, category=category, tags=tags, description=description, override=override)


def register_metric(
    name: str,
    *,
    category: str = "metric",
    tags: tuple[str, ...] | list[str] = (),
    description: str = "",
    override: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a metric function."""
    return METRIC_REGISTRY.register(name, category=category, tags=tags, description=description, override=override)


@dataclass(frozen=True)
class PluginLoadResult:
    """Summary of entry-point plugin discovery."""

    loaded: tuple[str, ...]
    failed: dict[str, str]


def discover_plugins(group: str = PLUGIN_ENTRY_POINT_GROUP) -> PluginLoadResult:
    """Load installed Python entry-point plugins for ``group``.

    Each entry point may point to a callable taking no arguments (which performs
    its own ``register_*`` calls) or a module whose import triggers decorators.
    """
    loaded: list[str] = []
    failed: dict[str, str] = {}
    eps = metadata.entry_points()
    selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
    for ep in selected:
        try:
            plugin = ep.load()
            if callable(plugin):
                plugin()
            loaded.append(ep.name)
        except Exception as exc:  # noqa: BLE001
            failed[ep.name] = f"{type(exc).__name__}: {exc}"
    return PluginLoadResult(loaded=tuple(loaded), failed=failed)


__all__ = [
    "BaseRunnerHook",
    "METRIC_REGISTRY",
    "MODEL_REGISTRY",
    "PLUGIN_ENTRY_POINT_GROUP",
    "PluginLoadResult",
    "Registry",
    "RegistryEntry",
    "RunnerHook",
    "TASK_REGISTRY",
    "discover_plugins",
    "register_metric",
    "register_model",
    "register_task",
]
