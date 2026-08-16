"""Regression tests for M0 Stage 0's crop-and-rebase step.

`_crop_and_convert_coordinates` shifts every tick label from original-image
coordinates into the cropped chart frame.  Downstream, Stage 1 and Stage 2 read
those rebased coordinates, so a defect here is silent: the pipeline keeps
running and reports coordinates measured against the wrong origin.

The y-label loop was previously nested inside the x-label loop, which made the
y list depend on there being at least one x label.  These tests pin the two
properties that nesting broke -- both axes are converted independently, and an
empty axis returns an empty list rather than raising.
"""

import numpy as np
import pytest

from recognizer.core.chart_extractor import ChartExtractor


def _label(text, value, left, top, right, bottom):
    return {
        'text': text,
        'value': value,
        'bbox': {'left': left, 'top': top, 'right': right, 'bottom': bottom},
        'center_x': (left + right) / 2,
        'center_y': (top + bottom) / 2,
        'original_type': 'x_axis_tick_label',
    }


@pytest.fixture
def crop():
    extractor = ChartExtractor()
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    bbox = {'left': 50, 'top': 30, 'right': 550, 'bottom': 370}
    return extractor, image, bbox


def test_both_axes_are_rebased_by_the_crop_origin(crop):
    extractor, image, bbox = crop
    x_raw = [_label('1', 1.0, 100, 380, 120, 395),
             _label('2', 2.0, 200, 380, 220, 395)]
    y_raw = [_label('10', 10.0, 20, 100, 40, 115),
             _label('20', 20.0, 20, 200, 40, 215),
             _label('30', 30.0, 20, 300, 40, 315)]

    _, bbox_tuple, x_conv, y_conv = extractor._crop_and_convert_coordinates(
        image, bbox, x_raw, y_raw)

    assert bbox_tuple == (50, 30, 500, 340)
    # Every label appears exactly once, whatever the other axis' count is.
    assert len(x_conv) == 2
    assert len(y_conv) == 3

    assert x_conv[0]['center_x'] == 110 - 50
    assert x_conv[0]['center_y'] == 387.5 - 30
    assert x_conv[0]['raw_center_x'] == 110
    assert x_conv[0]['raw_center_y'] == 387.5
    assert x_conv[0]['bbox'] == {'left': 50, 'top': 350,
                                 'right': 70, 'bottom': 365}

    assert y_conv[2]['center_x'] == 30 - 50
    assert y_conv[2]['center_y'] == 307.5 - 30
    assert y_conv[2]['value'] == 30.0


def test_no_x_labels_returns_empty_lists_not_unbound_error(crop):
    """The nested-loop defect's failure mode, pinned."""
    extractor, image, bbox = crop
    y_raw = [_label('10', 10.0, 20, 100, 40, 115)]

    _, _, x_conv, y_conv = extractor._crop_and_convert_coordinates(
        image, bbox, [], y_raw)

    assert x_conv == []
    assert len(y_conv) == 1
    assert y_conv[0]['center_y'] == 107.5 - 30


def test_no_y_labels_returns_empty_list(crop):
    extractor, image, bbox = crop
    x_raw = [_label('1', 1.0, 100, 380, 120, 395)]

    _, _, x_conv, y_conv = extractor._crop_and_convert_coordinates(
        image, bbox, x_raw, [])

    assert len(x_conv) == 1
    assert y_conv == []


def test_crop_bounds_are_clamped_to_the_image(crop):
    extractor, image, _ = crop
    oversized = {'left': -20, 'top': -10, 'right': 9000, 'bottom': 9000}

    chart_image, bbox_tuple, _, _ = extractor._crop_and_convert_coordinates(
        image, oversized, [], [])

    assert bbox_tuple == (0, 0, 600, 400)
    assert chart_image.shape[:2] == (400, 600)
