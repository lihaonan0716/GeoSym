"""Contract tests for the two single-source-of-truth constructors in M0.

Both were extracted from inline literals so that offline diagnostics could
record exactly what a run sends, instead of keeping a hand-transcribed second
copy. That only works if the extraction is faithful and the copy is real, so
those two properties are asserted here rather than assumed:

  * `build_request_body` is what `ocr_image` actually posts, and mutating its
    result cannot change what the next call sends;
  * `build_m0_stage_config` reproduces the configuration the pipeline ran
    under, field by field, and `skip_enhancement` is its only variable.
"""

import copy
from unittest.mock import patch

import pytest

from evaluator.adapters.recognizer import (
    M0_STAGE_CONFIG, build_m0_stage_config,
)
from recognizer.mathpix import mathpix_image_ocr as mx


# The configuration the M0 pipeline was evaluated under, transcribed here ONCE
# and deliberately: a test that imported the constant it is checking would pass
# no matter how the constant changed. Any edit to `M0_STAGE_CONFIG` must be
# mirrored here consciously, which is the point -- historical numbers were
# produced under these exact values.
EXPECTED_STAGE_CONFIG = {
    "stage0": {
        "horizontal_tolerance": 1.5,
        "vertical_tolerance": 1.5,
        "interval_percentile": 50,
    },
    "stage05": {"enabled": False},
    "stage1": {
        "enable_curve_erase": True,
        "enable_dashed_line_connection": True,
        "enable_perspective_correction": True,
        "enable_rotation_correction": True,
        "rotation_hough_threshold": 60,
        "rotation_min_line_length": 60,
        "rotation_max_line_gap": 10,
        "rotation_angle_threshold": 5,
        "grid_hough_threshold": 65,
        "grid_min_line_length": 45,
        "grid_max_line_gap": 10,
        "grid_angle_threshold": 5,
        "merge_distance_threshold": 20,
        "dashed_gap_threshold": 10,
        "dashed_collinear_threshold": 3,
        "curve_saturation_threshold": 90,
        "curve_value_threshold": 60,
        "curve_feather_size": 3,
    },
    "stage2": {
        "tolerance": 3,
        "enable_morph_connect": True,
        "morph_kernel_size": 7,
        "morph_iterations": 2,
        "color_saturation_threshold": 90,
        "color_value_threshold": 60,
        "integer_approximation_threshold": 0.1,
    },
}


def test_request_body_carries_the_documented_chart_options():
    body = mx.build_request_body("data:image/jpeg;base64,AAAA")

    assert body["src"] == "data:image/jpeg;base64,AAAA"
    # These two are the documented options for parsing charts and the text
    # inside them; losing either silently turns chart OCR into plain text OCR.
    assert body["include_line_data"] is True
    assert body["include_diagram_text"] is True
    assert body["formats"] == ["text", "data", "html"]


def test_request_body_is_deep_copied_per_call():
    first = mx.build_request_body("a")
    first["formats"].append("mutated")
    first["data_options"]["include_latex"] = "mutated"

    second = mx.build_request_body("b")

    assert second["formats"] == ["text", "data", "html"]
    assert second["data_options"]["include_latex"] is True
    assert mx.MATHPIX_REQUEST_OPTIONS["formats"] == ["text", "data", "html"]
    assert mx.MATHPIX_REQUEST_OPTIONS["data_options"]["include_latex"] is True


def test_ocr_image_posts_exactly_the_built_body():
    """The diagnostic's premise: recorded body == transmitted body."""
    src = "data:image/jpeg;base64,ZZZZ"
    sent = {}

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"line_data": []}

    def _fake_post(url, headers=None, json=None, timeout=None):
        sent["url"] = url
        sent["json"] = json
        return _Response()

    with patch.object(mx, "_load_mathpix_credentials",
                      return_value=("id", "key")), \
            patch.object(mx.requests, "post", _fake_post):
        mx.MathpixImageOCRTool().ocr_image(src)

    assert sent["url"] == mx.MATHPIX_ENDPOINT
    assert sent["json"] == mx.build_request_body(src)


def test_stage_config_matches_the_evaluated_values_field_by_field():
    config = build_m0_stage_config(skip_enhancement=True)

    assert config == EXPECTED_STAGE_CONFIG
    for stage, params in EXPECTED_STAGE_CONFIG.items():
        assert set(config[stage]) == set(params), stage
        for key, value in params.items():
            assert config[stage][key] == value, f"{stage}.{key}"
            assert type(config[stage][key]) is type(value), f"{stage}.{key}"


@pytest.mark.parametrize("skip", [True, False])
def test_skip_enhancement_is_the_only_thing_that_varies(skip):
    config = build_m0_stage_config(skip_enhancement=skip)

    assert config["stage05"]["enabled"] is (not skip)
    without_stage05 = {k: v for k, v in config.items() if k != "stage05"}
    expected = {k: v for k, v in EXPECTED_STAGE_CONFIG.items()
                if k != "stage05"}
    assert without_stage05 == expected


def test_stage_config_callers_cannot_mutate_the_module_constant():
    pristine = copy.deepcopy(M0_STAGE_CONFIG)

    config = build_m0_stage_config()
    config["stage1"]["grid_hough_threshold"] = 999
    config["stage05"]["enabled"] = "mutated"

    assert M0_STAGE_CONFIG == pristine
    assert build_m0_stage_config()["stage1"]["grid_hough_threshold"] == 65
