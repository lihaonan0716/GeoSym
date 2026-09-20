"""The evidence interface M0 implementations are substitutable through.

Three independent extractors exist in this repository -- the grid-based
recognizer, the OCR-ruler presenter, and the PlotQA mark chain. Until now they
shared an idea but no type: each returned its own shape, none imported another,
and nothing in the codebase could state which of them applies to a given chart.
"Component substitution" was a claim about the architecture that the
architecture did not support.

This package is that type. It adds a layer; it changes no algorithm. Every
provider wraps an existing implementation and calls it unmodified, so no
historical number moves.

Sub-modules are imported lazily. The providers pull in OpenCV, OCR backends and
the recognizer pipeline, and a caller that only needs the dataclasses should not
pay for those.
"""

from __future__ import annotations

__all__ = [
    "Applicability",
    "ChartEvidenceRequest",
    "EvidenceBundle",
    "EvidenceItem",
    "EvidenceProvider",
    "CleanGridProvider",
    "OcrRulerProvider",
    "PlotqaMarkChainProvider",
    "available_providers",
    "select_provider",
]

_LAZY_MAP = {
    "Applicability": (".interface", "Applicability"),
    "ChartEvidenceRequest": (".interface", "ChartEvidenceRequest"),
    "EvidenceBundle": (".interface", "EvidenceBundle"),
    "EvidenceItem": (".interface", "EvidenceItem"),
    "EvidenceProvider": (".interface", "EvidenceProvider"),
    "CleanGridProvider": (".providers", "CleanGridProvider"),
    "OcrRulerProvider": (".providers", "OcrRulerProvider"),
    "PlotqaMarkChainProvider": (".providers", "PlotqaMarkChainProvider"),
    "available_providers": (".providers", "available_providers"),
    "select_provider": (".providers", "select_provider"),
}


def __getattr__(name: str):
    if name in _LAZY_MAP:
        import importlib

        module_path, attr = _LAZY_MAP[name]
        value = getattr(importlib.import_module(module_path, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
