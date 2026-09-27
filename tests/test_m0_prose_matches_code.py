"""The executed M0 must keep matching what the paper says about it.

The manuscript described M0 as a four-stage pipeline beginning with "OCR is
used to identify axis tick labels", using a probabilistic Hough transform over
Canny/CLAHE/black-hat edge maps, perspective correction, greedy OCR-to-gridline
matching, a least-squares ruler cross-check, and a six-intersection minimum.

The main condition `E5_raw` executes none of those. Every paid run wraps its
loop in `clean_grid_only_m0()`, routing M0 to `extract_clean_cartesian_grid`,
whose only imports are cv2 and numpy. Five of six audited claims were
contradicted, and the drift survived three frozen plans and four paid runs
because checking it required reading both the prose and the code.

These tests pin the facts the prose must respect, so a reintroduced false claim
fails here instead of reaching a reviewer.
"""

import pytest

# Dev-local script (`scripts/*` is gitignored): skip rather than abort the
# suite's collection in a clean clone.
verifier = pytest.importorskip("scripts.verify_m0_prose_matches_code")


class TestExecutedPathUsesNoOcr:
    def test_the_grid_extractor_never_calls_ocr(self):
        """The whole point: on an integer grid the lattice IS the scale."""
        assert verifier.static_claims()["calls_ocr"] is False

    def test_entry_point_is_the_clean_grid_extractor_not_chart_extractor(self):
        """ChartExtractor is the paid Mathpix route; E5_raw must not name it."""
        assert verifier.EXECUTED_M0.endswith("extract_clean_cartesian_grid")
        assert "ChartExtractor" not in verifier.EXECUTED_M0


class TestClaimsThatWereFalse:
    @pytest.mark.parametrize("key", [
        "uses_hough", "uses_canny", "uses_clahe", "uses_blackhat",
        "uses_perspective_or_rotation_correction", "uses_least_squares_ruler",
    ])
    def test_absent_machinery_stays_absent(self, key):
        assert verifier.static_claims()[key] is False


class TestClaimsThatHeld:
    def test_curve_isolation_really_uses_hsv(self):
        assert verifier.static_claims()["uses_hsv_thresholding"] is True

    def test_scale_comes_from_median_lattice_spacing(self):
        facts = verifier.static_claims()
        assert facts["derives_scale_from_lattice_spacing"] is True
        assert facts["uses_projection_clustering"] is True


class TestVerifierReportsDriftHonestly:
    def test_clean_repository_reports_consistent(self):
        assert verifier.run(per_family=1)["status"] == "consistent"

    def test_a_violated_claim_is_named_not_summarised_away(self, monkeypatch):
        monkeypatch.setattr(verifier, "static_claims",
                            lambda: {**_baseline(), "calls_ocr": True})
        results = verifier.run(per_family=1)
        assert results["status"] == "prose_contradicts_code"
        assert results["violations"] >= 1
        violated = [a for a in results["audit"] if a["status"] == "VIOLATED"]
        assert any(a["fact_key"] == "calls_ocr" for a in violated)

    def test_report_records_zero_paid_calls(self):
        results = verifier.run(per_family=1)
        assert results["paid_calls_executed"] == 0
        assert results["actual_ocr_http_attempts"] == 0


def _baseline():
    return {
        "calls_ocr": False, "uses_hough": False, "uses_canny": False,
        "uses_clahe": False, "uses_blackhat": False,
        "uses_perspective_or_rotation_correction": False,
        "uses_least_squares_ruler": False, "uses_hsv_thresholding": True,
        "uses_projection_clustering": True,
        "derives_scale_from_lattice_spacing": True,
    }


class TestSixIntersectionThresholdDoesNotExist:
    def test_extractor_succeeds_below_six_intersections(self):
        """The manuscript claimed a six-intersection minimum. cubic returns 3."""
        measured = verifier.measured_claims(per_family=1)
        if not measured.get("available"):
            pytest.skip(measured.get("reason", "dataset unavailable"))
        assert measured["min_intersections_among_successes"] < 6
        assert measured["enforces_six_intersection_minimum"] is False

    def test_every_family_extracts_successfully(self):
        measured = verifier.measured_claims(per_family=1)
        if not measured.get("available"):
            pytest.skip(measured.get("reason", "dataset unavailable"))
        assert measured["successes"] == measured["samples"]
