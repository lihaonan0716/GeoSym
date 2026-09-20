"""The three M0 implementations, expressed through one interface.

Each provider is a translation layer over code that already exists. None
re-implements extraction, changes a threshold, or reorders a pipeline: the
wrapped functions are called exactly as their current callers call them, so
every historical number stays reproducible.

One property is worth stating up front because it constrains how these may be
used. ``requires_paid_ocr`` is not documentation -- :func:`available_providers`
filters on it, so a caller sweeping "every provider" over a roster cannot
silently bill a Mathpix call per image. The grid and PlotQA paths are free (pure
OpenCV, and frozen OCR boxes read from disk respectively); the ruler path is
not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluator.evidence.interface import (
    Applicability,
    ApplicabilityReport,
    ChartEvidenceRequest,
    EvidenceBundle,
    EvidenceItem,
    unsupported_bundle,
)


class CleanGridProvider:
    """Curve-grid intersections from a regular Cartesian grid.

    Wraps ``recognizer.core.clean_grid_fallback.extract_clean_cartesian_grid``,
    the pure-OpenCV path of the function-specialised recognizer. The full
    recognizer pipeline is deliberately not wrapped here: it calls Mathpix,
    which would make an ordinary provider sweep a paid operation.

    Its precondition is the sharpest of the three, and it is the reason this
    interface exists. The wrapped extractor requires at least three grid lines
    on each axis and rejects the chart otherwise, so gridlines are not a
    preference of this implementation -- they are its domain boundary.
    """

    name = "clean_grid"
    domain = "cartesian charts with a regular visible grid and integer lattice"
    item_semantics = "curve_grid_intersection"
    requires_paid_ocr = False

    #: Mirrors ``clean_grid_fallback.extract_clean_cartesian_grid``, which
    #: returns ``clean_grid_structure_not_detected`` below this count.
    MIN_GRID_LINES_PER_AXIS = 3

    def applicability(self, request: ChartEvidenceRequest) -> ApplicabilityReport:
        """Detect the grid, without extracting anything from it.

        Runs the same detector the extractor runs and reads its diagnostics, so
        the verdict cannot disagree with what extraction would do. Restating the
        grid-counting logic here would create a second source of truth that
        drifts.
        """
        import cv2

        from recognizer.core.clean_grid_fallback import extract_clean_cartesian_grid

        image = cv2.imread(str(request.image_path))
        if image is None:
            return ApplicabilityReport(
                Applicability.UNSUPPORTED,
                "image_unreadable",
                {"image_path": str(request.image_path)},
            )

        extraction = extract_clean_cartesian_grid(image)
        diagnostics = dict(extraction.result.get("diagnostics") or {})
        grid_x = int(diagnostics.get("grid_x_count") or 0)
        grid_y = int(diagnostics.get("grid_y_count") or 0)
        if grid_x < self.MIN_GRID_LINES_PER_AXIS or grid_y < self.MIN_GRID_LINES_PER_AXIS:
            return ApplicabilityReport(
                Applicability.UNSUPPORTED,
                "insufficient_grid_lines",
                diagnostics,
            )
        if diagnostics.get("axis_x") is None or diagnostics.get("axis_y") is None:
            return ApplicabilityReport(
                Applicability.UNSUPPORTED, "axes_not_located", diagnostics
            )
        return ApplicabilityReport(Applicability.SUPPORTED, "grid_detected", diagnostics)

    def extract(self, request: ChartEvidenceRequest) -> EvidenceBundle:
        import cv2

        from recognizer.core.clean_grid_fallback import extract_clean_cartesian_grid

        report = self.applicability(request)
        if not report.supported:
            return unsupported_bundle(self.name, report, item_semantics=self.item_semantics)

        image = cv2.imread(str(request.image_path))
        extraction = extract_clean_cartesian_grid(image)
        payload = extraction.result.get("result") or {}
        intersections = payload.get("intersections") or []

        items = [
            EvidenceItem(
                # ``real_*`` are already chart-space values here: this extractor
                # works in benchmark units, where the grid lattice *is* the
                # scale, so no pixel-to-value fit is involved.
                value_x=_float_or_none(point.get("real_x_rounded", point.get("real_x"))),
                value_y=_float_or_none(point.get("real_y_rounded", point.get("real_y"))),
                pixel_x=_float_or_none(point.get("pixel_x")),
                pixel_y=_float_or_none(point.get("pixel_y")),
                item_id=f"grid_{index:03d}",
                extra={"type": point.get("type"), "line_types": point.get("line_types")},
            )
            for index, point in enumerate(intersections)
        ]

        return EvidenceBundle(
            provider=self.name,
            applicability=report,
            items=items,
            item_semantics=self.item_semantics,
            axis_model=dict(payload.get("grid_mapping") or {}),
            succeeded=bool(extraction.success),
            failure_reason=extraction.result.get("error"),
            diagnostics=dict(extraction.result.get("diagnostics") or {}),
        )


class OcrRulerProvider:
    """Measured values at caller-supplied pixels, via an OCR-fitted ruler.

    Wraps ``presenter.ruler.build_ruler`` and ``presenter.measure.measure_points``
    unchanged.

    Two properties separate it from the others. It **needs located points**: it
    measures where it is told to look and does not find marks itself, so a
    request without ``located_points`` is out of domain rather than empty. And
    it **calls Mathpix**, so it is excluded from free sweeps by default.
    """

    name = "ocr_ruler"
    domain = "real-world charts with readable axis tick labels and no integer grid"
    item_semantics = "measured_point_at_located_pixel"
    requires_paid_ocr = True

    def applicability(self, request: ChartEvidenceRequest) -> ApplicabilityReport:
        """Check only what can be checked without paying for OCR.

        Whether the ticks are readable is exactly what the paid call decides, so
        this returns ``UNKNOWN`` rather than guessing. Reporting ``SUPPORTED``
        here would claim a fit that has not been attempted.
        """
        if not request.image_path.exists():
            return ApplicabilityReport(
                Applicability.UNSUPPORTED,
                "image_missing",
                {"image_path": str(request.image_path)},
            )
        if not request.located_points:
            return ApplicabilityReport(
                Applicability.UNSUPPORTED,
                "no_located_points_supplied",
                {"note": "this provider measures at given pixels; it does not find marks"},
            )
        return ApplicabilityReport(
            Applicability.UNKNOWN,
            "tick_readability_requires_paid_ocr",
            {"requires_paid_ocr": True},
        )

    def extract(self, request: ChartEvidenceRequest) -> EvidenceBundle:
        """Run the ruler and measure. **Spends one paid OCR call per image.**

        Guarded by ``options["allow_paid_ocr"]`` so the call cannot happen by
        accident -- a sweep that forgot this provider is paid gets a failed
        bundle, not a bill.
        """
        report = self.applicability(request)
        if report.verdict is Applicability.UNSUPPORTED:
            return unsupported_bundle(self.name, report, item_semantics=self.item_semantics)

        if not request.options.get("allow_paid_ocr"):
            return EvidenceBundle(
                provider=self.name,
                applicability=report,
                item_semantics=self.item_semantics,
                succeeded=False,
                failure_reason="paid_ocr_not_authorised",
                diagnostics={"hint": "set options['allow_paid_ocr'] = True to permit one Mathpix call"},
            )

        from presenter.measure import measure_points
        from presenter.ruler import build_ruler

        ruler = build_ruler(str(request.image_path))
        if not ruler.success:
            return EvidenceBundle(
                provider=self.name,
                applicability=report,
                item_semantics=self.item_semantics,
                succeeded=False,
                failure_reason=str(ruler.diagnostics.get("error") or "ruler_build_failed"),
                diagnostics=dict(ruler.diagnostics),
            )

        measured = measure_points(ruler, list(request.located_points or []))
        items = [
            EvidenceItem(
                value_x=_float_or_none(point.computed_x),
                value_y=_float_or_none(point.computed_y),
                pixel_x=point.pixel_x,
                pixel_y=point.pixel_y,
                item_id=f"ruler_{index:03d}",
                extra={
                    "role": point.role,
                    "description": point.description,
                    "computation_method": point.computation_method,
                },
            )
            for index, point in enumerate(measured)
        ]

        return EvidenceBundle(
            provider=self.name,
            applicability=report,
            items=items,
            item_semantics=self.item_semantics,
            axis_model={
                "x_scale_type": getattr(ruler.x_axis, "scale_type", None),
                "y_scale_type": getattr(ruler.y_axis, "scale_type", None),
                "chart_type": ruler.chart_type,
            },
            succeeded=True,
            diagnostics=dict(ruler.diagnostics),
        )


class PlotqaMarkChainProvider:
    """Plotted-mark values from the three-stage PlotQA M0' chain.

    Wraps ``evaluator.reasoning.plotqa_m0_chain.build_m0_prime_record``
    unchanged: category axis, then ruler refit, then mark admission.

    It takes a pre-extracted record plus frozen OCR boxes rather than an image,
    because that is how the chain is actually driven -- the boxes are read from
    disk, and re-running OCR would break reproduction of the frozen artifacts.
    Its notion of an evidence item is the broadest of the three: a mark can
    carry a series label and a category label, which neither other provider
    produces.
    """

    name = "plotqa_mark_chain"
    domain = "bar/dot/line charts with OCR-readable value ticks and no plotted grid"
    item_semantics = "plotted_mark_value"
    requires_paid_ocr = False

    def applicability(self, request: ChartEvidenceRequest) -> ApplicabilityReport:
        """Check the inputs, then ask the chain whether the axis is admissible.

        The verdict comes from running the chain's own refit rather than
        re-deriving a tick criterion, for the same reason as in
        :class:`CleanGridProvider`: one source of truth.
        """
        if request.record is None:
            return ApplicabilityReport(
                Applicability.UNSUPPORTED,
                "no_extracted_record_supplied",
                {"note": "this provider consumes an M0 record, not a raw image"},
            )
        if request.ocr_row is None:
            return ApplicabilityReport(
                Applicability.UNSUPPORTED,
                "no_ocr_row_supplied",
                {"note": "frozen OCR boxes are an input; this provider never calls OCR"},
            )

        from evaluator.reasoning.plotqa_m0_chain import build_m0_prime_record

        chained = build_m0_prime_record(request.record, request.ocr_row)
        refit = chained.get("ruler_refit") or {}
        if not refit.get("admissible"):
            diagnostics: Dict[str, Any] = {"tick_count": refit.get("tick_count")}
            reason = str(refit.get("reason") or "value_axis_not_admissible")
            # The chain collects ticks in a band outside the plot-area edge,
            # because a framed statistical chart prints them there. A Cartesian
            # plot prints them along axes through the origin instead, so its
            # labels sit mid-image and none land in the band. Both look like
            # "insufficient_ticks" from inside the refit, but only the second is
            # a statement about chart structure, and a reviewer needs to be able
            # to tell a domain boundary from a bad OCR read.
            if reason == "insufficient_ticks" and not refit.get("tick_count"):
                interior = _interior_tick_labels(request)
                if interior:
                    diagnostics["interior_numeric_labels"] = interior
                    reason = "ticks_on_interior_axes_not_plot_frame"
            return ApplicabilityReport(Applicability.UNSUPPORTED, reason, diagnostics)
        return ApplicabilityReport(
            Applicability.SUPPORTED,
            "value_axis_fitted_from_ocr_ticks",
            {"tick_count": refit.get("tick_count")},
        )

    def extract(self, request: ChartEvidenceRequest) -> EvidenceBundle:
        if request.record is None or request.ocr_row is None:
            return unsupported_bundle(
                self.name, self.applicability(request), item_semantics=self.item_semantics
            )

        from evaluator.reasoning.plotqa_m0_chain import build_m0_prime_record

        report = self.applicability(request)
        chained = build_m0_prime_record(request.record, request.ocr_row)
        points = list(chained.get("measured_points") or [])

        items = [
            EvidenceItem(
                # Only one axis carries a value here: the other is categorical,
                # which is why ``category_label`` exists as a field at all.
                value_y=_float_or_none(point.get("measured_value")),
                pixel_x=_float_or_none((point.get("bbox") or [None])[0]),
                series_label=point.get("series_label"),
                category_label=point.get("category_label"),
                item_id=str(point.get("fact_id") or f"mark_{index:03d}"),
                extra={"value_status": point.get("value_status")},
            )
            for index, point in enumerate(points)
        ]

        # Succeeded means the chain ran, not that the axis was admissible: an
        # inadmissible axis yields items whose values are all None, which is a
        # correct refusal to guess rather than a failure to run.
        return EvidenceBundle(
            provider=self.name,
            applicability=report,
            items=items,
            item_semantics=self.item_semantics,
            axis_model=dict(chained.get("ruler_refit") or {}),
            succeeded=True,
            diagnostics={
                "chain_version": chained.get("m0_prime_chain"),
                "evidence_gaps": chained.get("evidence_gaps") or [],
            },
        )


_PROVIDERS = (CleanGridProvider, OcrRulerProvider, PlotqaMarkChainProvider)


def available_providers(*, include_paid: bool = False) -> List[Any]:
    """Instantiate the providers a sweep may use.

    Paid providers are excluded unless asked for, so "run every provider" is a
    free operation by default.
    """
    return [
        provider()
        for provider in _PROVIDERS
        if include_paid or not provider.requires_paid_ocr
    ]


def select_provider(
    request: ChartEvidenceRequest, *, include_paid: bool = False
) -> Optional[Any]:
    """Return the first provider whose preconditions this chart satisfies.

    Order is declaration order, not a ranking: this picks an applicable
    provider, and deliberately does not claim to pick the best one. Providers
    returning ``UNKNOWN`` are skipped, because selecting on an unverified
    precondition is how an untested path gets reported as a supported one.
    """
    for provider in available_providers(include_paid=include_paid):
        if provider.applicability(request).supported:
            return provider
    return None


def _interior_tick_labels(request: ChartEvidenceRequest) -> int:
    """Count numeric OCR labels sitting well inside the plot area.

    Diagnostic only -- it explains a refusal, never grants one. A Cartesian plot
    draws its axes through the origin, so its tick labels cluster near the
    middle of the plot area rather than along its frame; a framed chart has
    essentially none there.
    """
    geometry = (request.record or {}).get("chart_geometry") or {}
    plot_area = geometry.get("plot_area_bbox")
    if not plot_area or len(plot_area) != 4:
        return 0
    left, top, right, bottom = (float(v) for v in plot_area)
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        return 0

    count = 0
    for box in (request.ocr_row or {}).get("ocr_boxes") or []:
        text = str(box.get("text", "")).strip()
        if not text or not _looks_numeric(text):
            continue
        bbox = box.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
        # "Interior" means beyond a generous margin on every side, so labels
        # merely sitting a little inside a mis-estimated frame do not count.
        if (
            left + width * 0.20 <= cx <= right - width * 0.20
            or top + height * 0.20 <= cy <= bottom - height * 0.20
        ):
            count += 1
    return count


def _looks_numeric(text: str) -> bool:
    cleaned = text.replace(",", "").replace("−", "-").strip()
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _float_or_none(value: Any) -> Optional[float]:
    """Coerce to float, or None when the value is absent or non-numeric.

    Non-numeric input is common and legitimate: the presenter formats
    categorical and datetime axes as strings, and a mark on an inadmissible
    axis carries None. Both must read as "no numeric value", never as 0.0.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
