"""Tests for the OCR backend ladder used by the PlotQA M0 rotated-tick probe.

PaddleOCR replaced EasyOCR as the project's OCR backend. The ladder keeps
EasyOCR reachable as a fallback for environments without a usable
``paddlepaddle`` wheel, so these tests pin two properties:

1. **Preference order and fallthrough.** PaddleOCR is tried first; a failure
   falls through to EasyOCR rather than aborting the probe.
2. **Recorded provenance.** Detector output differs between backends, so every
   result states which one produced it. Results are not poolable across
   backends, and a silent backend swap would corrupt any tick-derived figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_plotqa_measured_m0 as m0  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_cached_readers():
    """The readers are cached on the function object; drop them between tests."""
    for attribute in ("_paddle_reader", "_easy_reader"):
        if hasattr(m0.rotated_ocr_ticks, attribute):
            delattr(m0.rotated_ocr_ticks, attribute)
    yield
    for attribute in ("_paddle_reader", "_easy_reader"):
        if hasattr(m0.rotated_ocr_ticks, attribute):
            delattr(m0.rotated_ocr_ticks, attribute)


def _poly(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _install_fake_paddleocr(monkeypatch, reader_class):
    """Register stub ``paddleocr`` and ``paddle`` modules in ``sys.modules``.

    ``importlib.util.find_spec`` raises on an already-imported module whose
    ``__spec__`` is None, so the stubs need a real ModuleSpec -- a bare
    ``types.ModuleType`` is not enough to stand in for an importable package.
    """
    import importlib.machinery

    for name, attributes in (("paddleocr", {"PaddleOCR": reader_class}), ("paddle", {})):
        module = type(sys)(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)


IMAGE = Path("unused-by-the-stubs.png")


class TestLadderOrdering:
    def test_paddleocr_is_preferred_and_easyocr_is_not_called(self, monkeypatch):
        calls = []

        def paddle(_path):
            calls.append("paddle")
            return [(_poly(0, 0, 10, 10), "2013", 0.99)], "ok"

        def easy(_path):
            calls.append("easy")
            return [(_poly(0, 0, 10, 10), "1999", 0.99)], "ok"

        monkeypatch.setattr(m0, "_rotated_ticks_paddleocr", paddle)
        monkeypatch.setattr(m0, "_rotated_ticks_easyocr", easy)

        ticks, diagnostics = m0.rotated_ocr_ticks(IMAGE)
        assert calls == ["paddle"]
        assert diagnostics["backend"] == "PaddleOCR"
        assert [tick.value for tick in ticks] == [2013.0]

    def test_falls_through_to_easyocr_when_paddleocr_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            m0, "_rotated_ticks_paddleocr", lambda _p: ([], "paddleocr_unavailable"))
        monkeypatch.setattr(
            m0, "_rotated_ticks_easyocr",
            lambda _p: ([(_poly(0, 0, 10, 10), "2013", 0.9)], "ok"))

        ticks, diagnostics = m0.rotated_ocr_ticks(IMAGE)
        assert diagnostics["backend"] == "EasyOCR"
        assert len(ticks) == 1
        assert diagnostics["backend_ladder"] == [
            {"backend": "PaddleOCR", "status": "paddleocr_unavailable"},
            {"backend": "EasyOCR", "status": "ok"},
        ]

    def test_paddleocr_exception_does_not_abort_the_probe(self, monkeypatch):
        monkeypatch.setattr(
            m0, "_rotated_ticks_paddleocr",
            lambda _p: ([], "paddleocr_failed:RuntimeError:cuda oom"))
        monkeypatch.setattr(
            m0, "_rotated_ticks_easyocr",
            lambda _p: ([(_poly(0, 0, 10, 10), "42", 0.9)], "ok"))

        ticks, diagnostics = m0.rotated_ocr_ticks(IMAGE)
        assert diagnostics["backend"] == "EasyOCR"
        assert "cuda oom" in diagnostics["backend_ladder"][0]["status"]
        assert len(ticks) == 1

    def test_reports_failure_when_every_backend_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            m0, "_rotated_ticks_paddleocr", lambda _p: ([], "paddleocr_unavailable"))
        monkeypatch.setattr(
            m0, "_rotated_ticks_easyocr", lambda _p: ([], "easyocr_unavailable"))

        ticks, diagnostics = m0.rotated_ocr_ticks(IMAGE)
        assert ticks == []
        assert diagnostics["status"] == "no_viable_rotated_ocr_backend"
        assert len(diagnostics["backend_ladder"]) == 2

    def test_empty_result_from_paddleocr_falls_through(self, monkeypatch):
        """A backend that runs but finds nothing must not shadow the fallback."""
        monkeypatch.setattr(
            m0, "_rotated_ticks_paddleocr", lambda _p: ([], "paddleocr_no_text"))
        monkeypatch.setattr(
            m0, "_rotated_ticks_easyocr",
            lambda _p: ([(_poly(0, 0, 10, 10), "7", 0.9)], "ok"))

        _ticks, diagnostics = m0.rotated_ocr_ticks(IMAGE)
        assert diagnostics["backend"] == "EasyOCR"


class TestTickParsing:
    def _run(self, monkeypatch, boxes):
        monkeypatch.setattr(m0, "_rotated_ticks_paddleocr", lambda _p: (boxes, "ok"))
        monkeypatch.setattr(m0, "_rotated_ticks_easyocr", lambda _p: ([], "unused"))
        return m0.rotated_ocr_ticks(IMAGE)

    def test_drops_low_confidence_boxes(self, monkeypatch):
        ticks, _ = self._run(monkeypatch, [(_poly(0, 0, 10, 10), "2013", 0.2)])
        assert ticks == []

    def test_drops_non_numeric_text(self, monkeypatch):
        ticks, _ = self._run(monkeypatch, [(_poly(0, 0, 10, 10), "Year", 0.99)])
        assert ticks == []

    def test_source_box_id_records_the_backend(self, monkeypatch):
        ticks, _ = self._run(monkeypatch, [(_poly(0, 0, 10, 10), "2013", 0.99)])
        assert ticks[0].source_box_id.startswith("rotated_paddleocr_")

    def test_malformed_polygon_is_skipped_not_fatal(self, monkeypatch):
        ticks, diagnostics = self._run(monkeypatch, [
            ("not-a-polygon", "2013", 0.99),
            (_poly(0, 0, 10, 10), "2014", 0.99),
        ])
        assert diagnostics["status"] == "ok"
        assert [tick.value for tick in ticks] == [2014.0]

    def test_bbox_is_the_polygon_extent(self, monkeypatch):
        ticks, _ = self._run(monkeypatch, [(_poly(5, 7, 25, 19), "2013", 0.99)])
        assert ticks[0].bbox == [5.0, 7.0, 25.0, 19.0]

    def test_multi_value_text_is_split_across_interpolated_boxes(self, monkeypatch):
        ticks, _ = self._run(monkeypatch, [(_poly(0, 0, 40, 10), "10 20", 0.99)])
        assert [tick.value for tick in ticks] == [10.0, 20.0]
        assert all("::" in tick.text for tick in ticks)


class TestDeviceDefault:
    def test_probe_defaults_to_cpu(self, monkeypatch):
        """A probe must not contend for VRAM with a running GPU benchmark."""
        monkeypatch.delenv(m0.ROTATED_OCR_DEVICE_ENV, raising=False)
        captured = {}

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def predict(self, _path):
                return []

        _install_fake_paddleocr(monkeypatch, FakePaddleOCR)

        m0._rotated_ticks_paddleocr(IMAGE)
        assert captured.get("device") == "cpu"

    def test_device_can_be_overridden_by_env(self, monkeypatch):
        monkeypatch.setenv(m0.ROTATED_OCR_DEVICE_ENV, "gpu:0")
        captured = {}

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def predict(self, _path):
                return []

        _install_fake_paddleocr(monkeypatch, FakePaddleOCR)

        m0._rotated_ticks_paddleocr(IMAGE)
        assert captured.get("device") == "gpu:0"

    def test_textline_orientation_is_enabled(self, monkeypatch):
        """Rotated axis labels are the entire point of this probe."""
        monkeypatch.delenv(m0.ROTATED_OCR_DEVICE_ENV, raising=False)
        captured = {}

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def predict(self, _path):
                return []

        _install_fake_paddleocr(monkeypatch, FakePaddleOCR)

        m0._rotated_ticks_paddleocr(IMAGE)
        assert captured.get("use_textline_orientation") is True


class TestSurveyDeviceFlag:
    def test_survey_accepts_a_device_override(self):
        from scripts.run_plotqa_openocr_survey import parse_args

        args = parse_args(["--plan", "p.json", "--output-dir", "out", "--device", "cpu"])
        assert args.device == "cpu"

    def test_survey_device_defaults_to_auto_detection(self):
        from scripts.run_plotqa_openocr_survey import parse_args

        args = parse_args(["--plan", "p.json", "--output-dir", "out"])
        assert args.device is None

    def test_paddleocr_is_first_in_the_backend_ladder(self):
        from scripts.run_plotqa_openocr_survey import BACKEND_ORDER

        assert BACKEND_ORDER[0] == "PaddleOCR"
        assert "EasyOCR" in BACKEND_ORDER
