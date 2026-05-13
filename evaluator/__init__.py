# Evaluator module for LLM and GeoSym testing
#
# All sub-modules are imported lazily to avoid pulling in heavy
# transitive dependencies (sympy ~48 MB, google API ~112 MB) at
# package-import time.  Callers that do ``from evaluator.xxx import
# yyy`` bypass __init__ entirely and get only what they need.

__all__ = [
    "BaseLLMAdapter",
    "RecognizerAdapter",
    "OpenAIAdapter",
    "ClaudeAdapter",
    "GeminiAdapter",
    "QwenVLAdapter",
    "KimiAdapter",
    "GLMAdapter",
    "compute_metrics",
    "compare_results",
    "aggregate_metrics",
    "compute_image_expr_metrics",
    "aggregate_expr_metrics",
    "parse_expression",
    "compute_r2",
    "compute_nrmse",
    "symbolic_equivalence",
    "classify_type",
    "match_curves",
]

_LAZY_MAP = {
    "BaseLLMAdapter":            (".adapters.base",        "BaseLLMAdapter"),
    "RecognizerAdapter":         (".adapters.recognizer",  "RecognizerAdapter"),
    "compute_metrics":           (".metrics.coordinate",   "compute_metrics"),
    "compare_results":           (".metrics.coordinate",   "compare_results"),
    "aggregate_metrics":         (".metrics.coordinate",   "aggregate_metrics"),
    "compute_image_expr_metrics":(".metrics.expression",   "compute_image_expr_metrics"),
    "aggregate_expr_metrics":    (".metrics.expression",   "aggregate_expr_metrics"),
    "parse_expression":          (".metrics.expression",   "parse_expression"),
    "compute_r2":                (".metrics.expression",   "compute_r2"),
    "compute_nrmse":             (".metrics.expression",   "compute_nrmse"),
    "symbolic_equivalence":      (".metrics.expression",   "symbolic_equivalence"),
    "classify_type":             (".metrics.expression",   "classify_type"),
    "match_curves":              (".metrics.expression",   "match_curves"),
    "OpenAIAdapter":             (".adapters.openai",      "OpenAIAdapter"),
    "ClaudeAdapter":             (".adapters.claude",      "ClaudeAdapter"),
    "GeminiAdapter":             (".adapters.gemini",      "GeminiAdapter"),
    "QwenVLAdapter":             (".adapters.qwen",        "QwenVLAdapter"),
    "KimiAdapter":               (".adapters.kimi",        "KimiAdapter"),
    "GLMAdapter":                (".adapters.glm",         "GLMAdapter"),
}


def __getattr__(name):
    if name in _LAZY_MAP:
        module_path, attr = _LAZY_MAP[name]
        import importlib
        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
