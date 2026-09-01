"""Small explicit registry mirroring lmms-eval's task/model resolution."""

from __future__ import annotations

from typing import Any


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._objects: dict[str, Any] = {}
        self._aliases: dict[str, str] = {}

    def register(self, name: str, value: Any, aliases: tuple[str, ...] = ()) -> None:
        if name in self._objects or name in self._aliases:
            raise ValueError(f"duplicate {self.kind}: {name}")
        self._objects[name] = value
        for alias in aliases:
            if alias in self._objects or alias in self._aliases:
                raise ValueError(f"duplicate {self.kind} alias: {alias}")
            self._aliases[alias] = name

    def resolve(self, name: str) -> Any:
        canonical = self._aliases.get(name, name)
        try:
            return self._objects[canonical]
        except KeyError as error:
            known = sorted([*self._objects, *self._aliases])
            raise KeyError(f"unknown {self.kind} {name!r}; known={known}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._objects))


TASKS = Registry("task")
MODELS = Registry("model")
METHODS = Registry("method")
EVALUATORS = Registry("evaluator")
