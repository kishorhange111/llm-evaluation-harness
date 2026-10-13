"""Model backends and a factory that builds one from a config dict."""

from __future__ import annotations

from typing import Any

from evalkit.models.base import Model
from evalkit.models.dummy import DummyModel


def build_model(spec: dict[str, Any]) -> Model:
    """Create a model from a config entry such as ``{"backend": "hf", "pretrained": "..."}``.

    Heavy backends are imported only when requested, so ``import evalkit`` stays
    fast and works on machines without torch.
    """
    spec = dict(spec)
    backend = spec.pop("backend")
    if backend == "hf":
        from evalkit.models.hf import HFModel

        return HFModel(**spec)
    if backend == "openai":
        from evalkit.models.openai_compat import OpenAICompatibleModel

        return OpenAICompatibleModel(**spec)
    if backend == "dummy":
        return DummyModel(**spec)
    raise ValueError(f"unknown backend {backend!r} (expected hf, openai or dummy)")


__all__ = ["Model", "DummyModel", "build_model"]
