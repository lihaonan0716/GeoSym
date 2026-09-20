"""Regression tests for the M0 evidence interface.

The interface exists to make "M0 implementations are substitutable" checkable
rather than asserted, so these tests guard the properties that claim rests on:
that providers declare their domain, refuse charts outside it for a stated
reason, and never substitute a guess for an unmeasured value.

Two properties get the most coverage because breaking them would be silent:
paid-OCR isolation (a leak bills real money per image) and the
unmeasured-vs-zero distinction (a leak turns a refusal to guess into a
measurement of zero).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluator.evidence import (
    Applicability,
    ChartEvidenceRequest,
    EvidenceBundle,
    EvidenceItem,
    EvidenceProvider,
    available_providers,
    select_provider,
)
from evaluator.evidence.interface import ApplicabilityReport, unsupported_bundle
from evaluator.evidence.providers import (
    CleanGridProvider,
    OcrRulerProvider,
    PlotqaMarkChainProvider,
    _float_or_none,
)

ROOT = Path(__file__).resolve().parents[1]
GEOSYM_IMAGE = ROOT / "dataset" / "images" / "linear-clean_render" / "linear-clean_render_0000.png"


class TestProtocolConformance:
    def test_every_provider_satisfies_the_protocol(self):
        for provider in available_providers(include_paid=True):
            assert isinstance(provider, EvidenceProvider)

    def test_every_provider_declares_a_domain_and_semantics(self):
        for provider in available_providers(include_paid=True):
            assert provider.domain.strip(), f"{provider.name} has no domain"
            assert provider.item_semantics.strip()

    def test_provider_names_are_unique(self):
        names = [p.name for p in available_providers(include_paid=True)]
        assert len(names) == len(set(names))


class TestPaidOcrIsolation:
    """A provider sweep must not be able to spend money by default."""

    def test_free_sweep_excludes_the_paid_provider(self):
        free = {p.name for p in available_providers()}
        assert OcrRulerProvider.name not in free

    def test_paid_provider_appears_only_when_requested(self):
        everything = {p.name for p in available_providers(include_paid=True)}
        assert OcrRulerProvider.name in everything

    def test_paid_extraction_refuses_without_authorisation(self, tmp_path):
        image = tmp_path / "chart.png"
        image.write_bytes(b"not really a png")
        request = ChartEvidenceRequest(
            image_path=image,
            located_points=[{"pixel_x": 10.0, "pixel_y": 20.0}],
        )
        bundle = OcrRulerProvider().extract(request)
        assert bundle.succeeded is False
        assert bundle.failure_reason == "paid_ocr_not_authorised"
        assert bundle.items == []

    def test_select_provider_never_returns_a_paid_one_by_default(self, tmp_path):
        image = tmp_path / "chart.png"
        image.write_bytes(b"x")
        request = ChartEvidenceRequest(
            image_path=image,
            located_points=[{"pixel_x": 1.0, "pixel_y": 2.0}],
        )
        chosen = select_provider(request)
        assert chosen is None or not chosen.requires_paid_ocr


class TestApplicabilityIsHonest:
    def test_unknown_is_distinct_from_unsupported(self, tmp_path):
        """Tick readability needs a paid call, so it must not be guessed."""
        image = tmp_path / "chart.png"
        image.write_bytes(b"x")
        report = OcrRulerProvider().applicability(
            ChartEvidenceRequest(
                image_path=image, located_points=[{"pixel_x": 1.0, "pixel_y": 1.0}]
            )
        )
        assert report.verdict is Applicability.UNKNOWN
        assert report.supported is False

    def test_missing_image_is_unsupported_not_a_crash(self, tmp_path):
        request = ChartEvidenceRequest(image_path=tmp_path / "absent.png")
        report = CleanGridProvider().applicability(request)
        assert report.verdict is Applicability.UNSUPPORTED
        assert report.reason == "image_unreadable"

    def test_plotqa_chain_declines_without_its_inputs(self, tmp_path):
        request = ChartEvidenceRequest(image_path=tmp_path / "chart.png")
        report = PlotqaMarkChainProvider().applicability(request)
        assert report.verdict is Applicability.UNSUPPORTED
        assert report.reason == "no_extracted_record_supplied"

    def test_ruler_provider_declines_without_located_points(self, tmp_path):
        image = tmp_path / "chart.png"
        image.write_bytes(b"x")
        report = OcrRulerProvider().applicability(ChartEvidenceRequest(image_path=image))
        assert report.verdict is Applicability.UNSUPPORTED
        assert report.reason == "no_located_points_supplied"

    def test_selection_skips_providers_reporting_unknown(self, tmp_path):
        """An unverified precondition must not be selected on."""
        image = tmp_path / "chart.png"
        image.write_bytes(b"x")
        request = ChartEvidenceRequest(
            image_path=image, located_points=[{"pixel_x": 1.0, "pixel_y": 1.0}]
        )
        assert select_provider(request, include_paid=True) is None


class TestUnmeasuredIsNotZero:
    """The interface's core honesty property: None never becomes 0.0."""

    @pytest.mark.parametrize("value", [None, "", "n/a", "2020-04", float("nan")])
    def test_non_numeric_values_do_not_become_zero(self, value):
        result = _float_or_none(value)
        assert result is None or result != 0.0

    def test_booleans_are_not_treated_as_numbers(self):
        assert _float_or_none(True) is None
        assert _float_or_none(False) is None

    def test_zero_survives_as_a_real_measurement(self):
        assert _float_or_none(0) == 0.0
        assert _float_or_none("0.0") == 0.0

    def test_item_without_values_is_not_measured(self):
        assert EvidenceItem().measured is False
        assert EvidenceItem(value_y=0.0).measured is True

    def test_measured_items_filters_unmeasured(self):
        bundle = EvidenceBundle(
            provider="x",
            applicability=ApplicabilityReport(Applicability.SUPPORTED, "ok"),
            items=[EvidenceItem(), EvidenceItem(value_y=3.0)],
        )
        assert len(bundle.items) == 2
        assert len(bundle.measured_items) == 1


class TestBundleSemantics:
    def test_success_is_not_inferred_from_item_count(self):
        """Zero items from a clean run differs from a failed extraction."""
        report = ApplicabilityReport(Applicability.SUPPORTED, "ok")
        empty_success = EvidenceBundle(provider="x", applicability=report, succeeded=True)
        assert empty_success.succeeded is True
        assert empty_success.items == []

    def test_unsupported_bundle_records_the_reason(self):
        report = ApplicabilityReport(Applicability.UNSUPPORTED, "insufficient_grid_lines")
        bundle = unsupported_bundle("clean_grid", report)
        assert bundle.succeeded is False
        assert "insufficient_grid_lines" in bundle.failure_reason


@pytest.mark.skipif(not GEOSYM_IMAGE.is_file(), reason="GeoSym-Bench dataset not present")
class TestAgainstRealCharts:
    """The substitutability claim, on actual images rather than fixtures."""

    def test_grid_provider_supports_a_gridded_chart(self):
        request = ChartEvidenceRequest(image_path=GEOSYM_IMAGE)
        report = CleanGridProvider().applicability(request)
        assert report.supported, report.reason

    def test_grid_provider_extracts_measured_items(self):
        bundle = CleanGridProvider().extract(ChartEvidenceRequest(image_path=GEOSYM_IMAGE))
        assert bundle.succeeded
        assert bundle.measured_items
        assert bundle.item_semantics == "curve_grid_intersection"

    def test_gridless_chart_is_declined_for_a_structural_reason(self, tmp_path):
        """A blank image has no grid, and the provider must say exactly that.

        This is the property the whole interface is for: being outside a
        provider's domain is a reported structural fact, not a silent empty
        result.
        """
        import numpy as np
        import cv2

        blank = np.full((400, 600, 3), 255, dtype=np.uint8)
        path = tmp_path / "blank.png"
        cv2.imwrite(str(path), blank)
        report = CleanGridProvider().applicability(ChartEvidenceRequest(image_path=path))
        assert report.verdict is Applicability.UNSUPPORTED
        assert report.reason == "insufficient_grid_lines"


class TestChartStructureIsDistinguishedFromBadInput:
    """A domain boundary must not read the same as a failed OCR read.

    The PlotQA chain gathers ticks in a band outside the plot frame. A Cartesian
    plot puts them on axes through the origin, so the chain finds none -- which
    is a fact about chart structure, not about OCR quality. Both surfaced as
    `insufficient_ticks` until this was separated, and a reviewer could not have
    told which had happened.
    """

    @staticmethod
    def _request(labels, plot_area=(100.0, 100.0, 900.0, 900.0)):
        return ChartEvidenceRequest(
            image_path=Path("unused.png"),
            record={
                "sample_id": "s",
                "chart_geometry": {"plot_area_bbox": list(plot_area)},
                "measured_points": [],
            },
            ocr_row={
                "sample_id": "s",
                "image_size": [1000, 1000],
                "ocr_boxes": [
                    {"box_id": f"b{i}", "text": text, "bbox": [cx - 8, cy - 8, cx + 8, cy + 8]}
                    for i, (text, cx, cy) in enumerate(labels)
                ],
            },
        )

    def test_interior_axis_labels_are_reported_as_a_structural_reason(self):
        # Numeric labels down the middle: a Cartesian plot's y axis.
        labels = [(str(v), 500.0, 200.0 + 60 * v) for v in range(1, 8)]
        report = PlotqaMarkChainProvider().applicability(self._request(labels))
        assert report.verdict is Applicability.UNSUPPORTED
        assert report.reason == "ticks_on_interior_axes_not_plot_frame"
        assert report.diagnostics["interior_numeric_labels"] >= 7

    def test_a_chart_with_no_labels_keeps_the_plain_reason(self):
        """No labels anywhere is a read failure, not a structural difference."""
        report = PlotqaMarkChainProvider().applicability(self._request([]))
        assert report.verdict is Applicability.UNSUPPORTED
        assert report.reason == "insufficient_ticks"
        assert "interior_numeric_labels" not in report.diagnostics

    def test_non_numeric_interior_text_does_not_trigger_the_reason(self):
        """Legend and title text sit inside the frame on ordinary charts."""
        labels = [("Bangladesh", 500.0, 300.0), ("Tonga", 500.0, 400.0)]
        report = PlotqaMarkChainProvider().applicability(self._request(labels))
        assert report.reason == "insufficient_ticks"

    def test_diagnostic_never_grants_applicability(self):
        """It explains a refusal; it must not turn one into a pass."""
        labels = [(str(v), 500.0, 200.0 + 60 * v) for v in range(1, 12)]
        report = PlotqaMarkChainProvider().applicability(self._request(labels))
        assert report.supported is False


class TestNoReliabilityScoringReenters:
    """M1/M2 retired 2026-09-19 (invariants 8-9); the interface must not host them."""

    def test_interface_exposes_no_scoring_or_selection_fields(self):
        forbidden = {"score", "reliability", "confidence_score", "tier", "rank", "weight"}
        for field_name in EvidenceItem.__dataclass_fields__:
            assert field_name not in forbidden
        for field_name in EvidenceBundle.__dataclass_fields__:
            assert field_name not in forbidden

    def test_providers_expose_no_scoring_method(self):
        for provider in available_providers(include_paid=True):
            for attribute in ("score_evidence", "select_evidence", "rank_items"):
                assert not hasattr(provider, attribute)
