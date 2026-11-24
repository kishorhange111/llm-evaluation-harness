"""Model backends and a factory that builds one from a config dict."""

from __future__ import annotations

from typing import Any

from evalkit.models.base import Model
from evalkit.models.dummy import DummyModel
from evalkit.registry import MODEL_REGISTRY, register_model


def _build_hf(**spec: Any) -> Model:
    from evalkit.models.hf import HFModel

    return HFModel(**spec)


def _build_openai(**spec: Any) -> Model:
    from evalkit.models.openai_compat import OpenAICompatibleModel

    return OpenAICompatibleModel(**spec)


def _build_dummy(**spec: Any) -> Model:
    return DummyModel(**spec)


for _name, _factory, _desc in (
    ("hf", _build_hf, "Local Hugging Face transformers model backend."),
    ("openai", _build_openai, "OpenAI-compatible HTTP inference server backend."),
    ("dummy", _build_dummy, "Deterministic offline mock model backend."),
):
    if _name not in MODEL_REGISTRY:
        MODEL_REGISTRY.register(_name, _factory, category="model", description=_desc)


def build_model(spec: dict[str, Any]) -> Model:
    """Create a model from a config entry such as ``{"backend": "hf", "pretrained": "..."}``.

    Heavy backends are imported only when requested, so ``import evalkit`` stays
    fast and works on machines without torch.
    """
    spec = dict(spec)
    backend = spec.pop("backend")
    factory = MODEL_REGISTRY.get(backend)
    return factory(**spec)


__all__ = ["Model", "DummyModel", "MODEL_REGISTRY", "build_model", "register_model"]
