"""
Handler functions for the coordinate recognizer Gradio UI.
All stage wrapper functions are extracted here for reusability.
"""

import cv2
import json
import numpy as np

from PIL import Image

import sys
from pathlib import Path

_exp_root = Path(__file__).resolve().parent.parent
if str(_exp_root) not in sys.path:
    sys.path.insert(0, str(_exp_root))

from i18n import t, get_lang  # noqa: E402


def generate_ocr_visualization(image, ocr_result):
    """Generate OCR visualization image with annotated elements."""
    if not ocr_result or 'line_data' not in ocr_result:
        return None

    vis_image = image.copy()

    type_colors = {
        'chart': (255, 0, 0),
        'x_axis_tick_label': (0, 255, 0),
        'y_axis_tick_label': (0, 0, 255),
        'model_label': (255, 255, 0),
        'legend_label': (255, 0, 255),
        'chart_info': (0, 255, 255),
        'text': (128, 128, 128),
        'default': (200, 200, 200)
    }

    for item in ocr_result['line_data']:
        item_type = item.get('type', 'default')
        text = item.get('text', '')
        cnt = item.get('cnt', [])

        if not cnt or len(cnt) < 4:
            continue

        color = type_colors.get(item_type, type_colors['default'])
        xs = [point[0] for point in cnt]
        ys = [point[1] for point in cnt]
        left, top = int(min(xs)), int(min(ys))
        right, bottom = int(max(xs)), int(max(ys))

        cv2.rectangle(vis_image, (left, top), (right, bottom), color, 2)

        label_text = f"{item_type}: {text}" if text else item_type
        font_scale = 0.5
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        text_x = max(0, left)
        text_y = max(text_height + 5, top - 5)

        cv2.rectangle(vis_image,
                      (text_x, text_y - text_height - 5),
                      (text_x + text_width, text_y),
                      color, -1)

        cv2.putText(vis_image, label_text, (text_x, text_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    legend_y = 30
    legend_items = [
        ('Chart', type_colors['chart']),
        ('X-axis', type_colors['x_axis_tick_label']),
        ('Y-axis', type_colors['y_axis_tick_label']),
        ('Model', type_colors['model_label']),
        ('Legend', type_colors['legend_label']),
        ('Info', type_colors['chart_info']),
    ]

    for idx, (name, color) in enumerate(legend_items):
        y_pos = legend_y + idx * 25
        cv2.rectangle(vis_image, (10, y_pos), (30, y_pos + 15), color, -1)
        cv2.putText(vis_image, name, (35, y_pos + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return vis_image


def extract_chart_handler(chart_extractor, problem_image,
                          horizontal_tolerance, vertical_tolerance,
                          interval_percentile, gemini_enabled):
    """Stage 0: Extract chart from problem image."""
    lang = get_lang()

    if problem_image is None:
        return None, None, t("rec_err_upload_first", lang), None, None, None, None, None, None

    try:
        if isinstance(problem_image, Image.Image):
            problem_image = np.array(problem_image)

        if len(problem_image.shape) == 3 and problem_image.shape[2] == 3:
            problem_image = cv2.cvtColor(problem_image, cv2.COLOR_RGB2BGR)

        result = chart_extractor.extract_chart_from_image(
            problem_image,
            horizontal_tolerance=horizontal_tolerance,
            vertical_tolerance=vertical_tolerance,
            interval_percentile=interval_percentile,
            gemini_enabled=gemini_enabled
        )

        debug_info_text = ""
        debug_image_rgb = None
        ocr_json_data = ""
        ocr_visual_image = None
        gemini_annotated_image = None
        gemini_comparison_text = ""

        if result.get('ocr_result'):
            ocr_json_data = json.dumps(result['ocr_result'], indent=2, ensure_ascii=False)

        if result.get('ocr_result'):
            ocr_visual_image = generate_ocr_visualization(problem_image, result['ocr_result'])
            if ocr_visual_image is not None:
                ocr_visual_image = cv2.cvtColor(ocr_visual_image, cv2.COLOR_BGR2RGB)

        if result.get('debug_info'):
            debug_info = result['debug_info']
            debug_lines = []
            debug_lines.append("=== Debug Info ===\n")

            params = debug_info.get('parameters', {})
            debug_lines.append(f"Tolerance params:")
            debug_lines.append(f"  Horizontal: {params.get('horizontal_tolerance', 'N/A')}")
            debug_lines.append(f"  Vertical: {params.get('vertical_tolerance', 'N/A')}")
            debug_lines.append(f"  Interval percentile: P{int(params.get('interval_percentile', 50))}\n")

            stage1 = debug_info.get('stage1_collection', {})
            debug_lines.append(f"Phase 1 - Candidate collection:")
            debug_lines.append(f"  Total candidates: {stage1.get('total_candidates', 0)}")
            debug_lines.append(f"  Type counts: {stage1.get('label_type_counts', {})}\n")

            stage2 = debug_info.get('stage2_clustering', {})
            debug_lines.append(f"Phase 2 - Position clustering:")
            debug_lines.append(f"  Numeric labels: {stage2.get('numeric_count', 0)}")
            debug_lines.append(f"  Horizontal clusters: {stage2.get('horizontal_clusters_count', 0)}")
            debug_lines.append(f"  Vertical clusters: {stage2.get('vertical_clusters_count', 0)}")
            debug_lines.append(f"  X-axis result: {stage2.get('x_axis_raw_count', 0)}")
            debug_lines.append(f"  Y-axis result: {stage2.get('y_axis_raw_count', 0)}\n")

            x_raw = stage2.get('x_axis_raw', [])
            if x_raw:
                debug_lines.append(f"  X-axis label details:")
                for label in x_raw:
                    debug_lines.append(f"    - value={label['value']}, text='{label['text']}', "
                                       f"type={label['original_type']}, "
                                       f"pos=({label['center_x']:.1f}, {label['center_y']:.1f})")
                debug_lines.append("")

            y_raw = stage2.get('y_axis_raw', [])
            if y_raw:
                debug_lines.append(f"  Y-axis label details:")
                for label in y_raw:
                    debug_lines.append(f"    - value={label['value']}, text='{label['text']}', "
                                       f"type={label['original_type']}, "
                                       f"pos=({label['center_x']:.1f}, {label['center_y']:.1f})")
                debug_lines.append("")

            stage3 = debug_info.get('stage3_filtering', {})
            debug_lines.append(f"Phase 3 - OCR filtering:")
            debug_lines.append(f"  X-axis after filter: {stage3.get('x_axis_filtered_count', 0)}")
            debug_lines.append(f"  Y-axis after filter: {stage3.get('y_axis_filtered_count', 0)}")

            debug_info_text = "\n".join(debug_lines)

        if result.get('debug_image') is not None:
            debug_image_rgb = cv2.cvtColor(result['debug_image'], cv2.COLOR_BGR2RGB)

        gemini_debug_info = result.get('gemini_debug_info', {'enabled': False})
        if gemini_debug_info.get('enabled'):
            if gemini_debug_info.get('annotated_image') is not None:
                gemini_annotated_image = cv2.cvtColor(gemini_debug_info['annotated_image'], cv2.COLOR_BGR2RGB)

            if gemini_debug_info.get('ocr_vs_gemini'):
                comparison_lines = []
                comparison_lines.append("=== OCR vs Gemini ===\n")
                if gemini_debug_info.get('success'):
                    comparison_lines.append("Gemini recognition successful\n")
                else:
                    comparison_lines.append(f"Gemini recognition failed: {gemini_debug_info.get('error', 'unknown')}\n")

                comparison_lines.append("| Label | OCR Text | OCR Value | Gemini Text | Gemini Value | Status |")
                comparison_lines.append("|-------|----------|-----------|-------------|--------------|--------|")

                for item in gemini_debug_info['ocr_vs_gemini']:
                    key = item['key']
                    ocr_text = item['ocr_text']
                    ocr_value = item['ocr_value']
                    gemini_text = item['gemini_text']
                    gemini_value = item.get('gemini_value', '-')

                    if not item.get('recognized', False):
                        status = "Not recognized"
                    elif gemini_value is None:
                        status = "Parse failed"
                    elif item.get('is_modified', False):
                        status = "Corrected"
                    else:
                        status = "Consistent"

                    comparison_lines.append(
                        f"| {key} | {ocr_text} | {ocr_value} | {gemini_text} | {gemini_value} | {status} |"
                    )

                gemini_comparison_text = "\n".join(comparison_lines)

        if not result['success']:
            error_msg = t("rec_err_chart_failed", lang).format(detail=result['error'])
            return None, None, error_msg, debug_info_text, debug_image_rgb, ocr_json_data, ocr_visual_image, gemini_annotated_image, gemini_comparison_text

        chart_image = result['chart_image']
        if chart_image is None or chart_image.size == 0:
            return None, None, t("rec_err_chart_empty", lang), debug_info_text, debug_image_rgb, ocr_json_data, ocr_visual_image, gemini_annotated_image, gemini_comparison_text

        chart_rgb = cv2.cvtColor(chart_image, cv2.COLOR_BGR2RGB)

        info_lines = []
        info_lines.append(t("rec_msg_chart_success", lang))
        info_lines.append(f"\nChart position: {result['chart_bbox']}")
        info_lines.append(f"Chart size: {chart_image.shape[1]}x{chart_image.shape[0]}")
        info_lines.append(f"\nX-axis tick labels: {len(result['x_tick_labels'])}")
        info_lines.append(f"Y-axis tick labels: {len(result['y_tick_labels'])}")

        if result['x_tick_labels']:
            x_values = [label['value'] for label in result['x_tick_labels']]
            info_lines.append(f"X-axis values: {x_values}")

        if result['y_tick_labels']:
            y_values = [label['value'] for label in result['y_tick_labels']]
            info_lines.append(f"Y-axis values: {y_values}")

        info_text = "\n".join(info_lines)

        stage0_data = {
            'chart_image_cv': chart_image,
            'x_tick_labels': result['x_tick_labels'],
            'y_tick_labels': result['y_tick_labels'],
            'ocr_result': result['ocr_result']
        }

        return chart_rgb, stage0_data, info_text, debug_info_text, debug_image_rgb, ocr_json_data, ocr_visual_image, gemini_annotated_image, gemini_comparison_text

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return None, None, f"Error: {str(e)}\n\n{error_detail}", "", None, None, None, None, ""


def ai_process_handler(image_enhancer, preprocessor, stage0_data,
                       enable_processing, selected_model, ai_prompt):
    """Stage 0.5: AI image enhancement."""
    lang = get_lang()

    if stage0_data is None:
        return None, None, None, t("rec_err_stage0_first", lang)

    try:
        original_image = stage0_data['chart_image_cv']

        if not enable_processing:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            info_text = t("rec_msg_ai_skipped", lang)

            processed_data = {
                'chart_image_cv': original_image,
                'x_tick_labels': stage0_data.get('x_tick_labels', []),
                'y_tick_labels': stage0_data.get('y_tick_labels', []),
                'ocr_result': stage0_data.get('ocr_result', {})
            }

            return original_rgb, original_rgb, processed_data, info_text

        if not ai_prompt or ai_prompt.strip() == "":
            return None, None, None, t("rec_err_enter_prompt", lang)

        result = image_enhancer.process_image(original_image, selected_model, ai_prompt)
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        if result['success']:
            processed_image = result['processed_image']

            print(f"\n[Stage 0.5] Resizing image to match original...")
            processed_image, ai_transform_params = preprocessor.resize_to_match_original(
                processed_image, original_image.shape[:2]
            )

            x_tick_labels = stage0_data.get('x_tick_labels', [])
            y_tick_labels = stage0_data.get('y_tick_labels', [])

            if ai_transform_params and (x_tick_labels or y_tick_labels):
                print(f"[Stage 0.5] Transforming tick labels...")
                if x_tick_labels:
                    x_tick_labels = preprocessor.transform_labels_for_ai_enhanced(
                        x_tick_labels, ai_transform_params
                    )
                if y_tick_labels:
                    y_tick_labels = preprocessor.transform_labels_for_ai_enhanced(
                        y_tick_labels, ai_transform_params
                    )

            processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

            info_lines = []
            model_display_name = image_enhancer.supported_models.get(selected_model, selected_model)
            info_lines.append(f"{model_display_name} processing successful!")
            info_lines.append(f"\nOriginal size: {original_image.shape[1]}x{original_image.shape[0]}")
            info_lines.append(f"AI output size: {result['processed_image'].shape[1]}x{result['processed_image'].shape[0]}")
            info_lines.append(f"Adjusted size: {processed_image.shape[1]}x{processed_image.shape[0]}")

            if result['response_text']:
                info_lines.append(f"\nModel response:")
                info_lines.append(result['response_text'][:500])

            info_text = "\n".join(info_lines)

            processed_data = {
                'chart_image_cv': processed_image,
                'x_tick_labels': x_tick_labels,
                'y_tick_labels': y_tick_labels,
                'ocr_result': stage0_data.get('ocr_result', {}),
                'ai_transform_params': ai_transform_params
            }

            return original_rgb, processed_rgb, processed_data, info_text
        else:
            info_lines = []
            model_display_name = image_enhancer.supported_models.get(selected_model, selected_model)
            info_lines.append(f"{model_display_name} processing failed, using original image.")
            info_lines.append(f"\nError: {result['error']}")

            if result['response_text']:
                info_lines.append(f"\nModel response:")
                info_lines.append(result['response_text'][:500])

            info_text = "\n".join(info_lines)

            processed_data = {
                'chart_image_cv': original_image,
                'x_tick_labels': stage0_data.get('x_tick_labels', []),
                'y_tick_labels': stage0_data.get('y_tick_labels', []),
                'ocr_result': stage0_data.get('ocr_result', {})
            }

            return original_rgb, original_rgb, processed_data, info_text

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()

        if stage0_data is not None and 'chart_image_cv' in stage0_data:
            original_image = stage0_data['chart_image_cv']
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            processed_data = {
                'chart_image_cv': original_image,
                'x_tick_labels': stage0_data.get('x_tick_labels', []),
                'y_tick_labels': stage0_data.get('y_tick_labels', []),
                'ocr_result': stage0_data.get('ocr_result', {})
            }

            error_msg = f"Error (fallback to original): {str(e)}\n\n{error_detail}"
            return original_rgb, original_rgb, processed_data, error_msg
        else:
            return None, None, None, f"Error: {str(e)}\n\n{error_detail}"


def stage1_process_handler(preprocessor, use_extracted_chart, stage0_data, gemini_data, manual_image,
                           enable_curve_erase, enable_dashed_connection, enable_perspective, enable_rotation,
                           curve_sat_thresh, curve_val_thresh, curve_feather,
                           dashed_gap_thresh, dashed_collinear_thresh,
                           rot_hough_thresh, rot_min_line_len, rot_max_gap, rot_angle_thresh,
                           grid_hough_thresh, grid_min_line_len, grid_max_gap, grid_angle_thresh,
                           merge_dist_thresh):
    """Stage 1: Image preprocessing."""
    lang = get_lang()

    if use_extracted_chart:
        if gemini_data is not None:
            print("[Stage 1] Using AI-processed image")
            image_cv = gemini_data['chart_image_cv']
            source_data = gemini_data
        elif stage0_data is not None:
            print("[Stage 1] Using Stage 0 chart")
            image_cv = stage0_data['chart_image_cv']
            source_data = stage0_data
        else:
            return None, None, None, None, None, t("rec_err_stage0_first", lang), None
    elif manual_image is not None:
        print("[Stage 1] Using manually uploaded image")
        image_cv = cv2.cvtColor(np.array(manual_image), cv2.COLOR_RGB2BGR)
        source_data = None
    else:
        return None, None, None, None, None, t("rec_err_extract_or_upload", lang), None

    result, intermediate = preprocessor.process(
        image_cv,
        enable_curve_erase=enable_curve_erase,
        enable_dashed_line_connection=enable_dashed_connection,
        enable_perspective_correction=enable_perspective,
        enable_rotation_correction=enable_rotation,
        x_tick_labels=source_data.get('x_tick_labels', []) if source_data else None,
        y_tick_labels=source_data.get('y_tick_labels', []) if source_data else None,
        curve_saturation_threshold=int(curve_sat_thresh),
        curve_value_threshold=int(curve_val_thresh),
        curve_feather_size=int(curve_feather),
        dashed_gap_threshold=int(dashed_gap_thresh),
        dashed_collinear_threshold=int(dashed_collinear_thresh),
        rotation_hough_threshold=int(rot_hough_thresh),
        rotation_min_line_length=int(rot_min_line_len),
        rotation_max_line_gap=int(rot_max_gap),
        rotation_angle_threshold=int(rot_angle_thresh),
        grid_hough_threshold=int(grid_hough_thresh),
        grid_min_line_length=int(grid_min_line_len),
        grid_max_line_gap=int(grid_max_gap),
        grid_angle_threshold=int(grid_angle_thresh),
        merge_distance_threshold=int(merge_dist_thresh)
    )

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    outputs = [result_rgb]

    if 'edges' in intermediate:
        outputs.append(intermediate['edges'])
    else:
        outputs.append(None)

    if 'detected_lines' in intermediate:
        outputs.append(cv2.cvtColor(intermediate['detected_lines'], cv2.COLOR_BGR2RGB))
    else:
        outputs.append(None)

    if 'dashed_connection' in intermediate:
        outputs.append(cv2.cvtColor(intermediate['dashed_connection'], cv2.COLOR_BGR2RGB))
    else:
        outputs.append(None)

    if 'merged_grid' in intermediate:
        outputs.append(cv2.cvtColor(intermediate['merged_grid'], cv2.COLOR_BGR2RGB))
    else:
        outputs.append(None)

    info_lines = []

    if 'perspective_matrix' in intermediate and intermediate['perspective_matrix'] is not None:
        info_lines.append("Applied perspective correction")
        if 'grid_corners' in intermediate:
            corners = intermediate['grid_corners']
            info_lines.append(f"  Detected corners: {len(corners)}")
    else:
        if enable_perspective:
            info_lines.append("Perspective correction failed (corner detection failed)")
        else:
            info_lines.append("Perspective correction disabled")

    if 'rotation_angle' in intermediate:
        info_lines.append(f"Detected tilt angle: {intermediate['rotation_angle']:.2f} deg")
    else:
        info_lines.append("No tilt correction")

    if 'merged_h_lines' in intermediate and 'merged_v_lines' in intermediate:
        info_lines.append(f"\nHorizontal grid lines: {len(intermediate['merged_h_lines'])}")
        info_lines.append(f"Vertical grid lines: {len(intermediate['merged_v_lines'])}")

    if 'h_gap_median' in intermediate:
        info_lines.append(f"\nHorizontal grid spacing (px):")
        info_lines.append(f"  Median(P50): {intermediate['h_gap_median']:.1f}")
        info_lines.append(f"  Min: {intermediate['h_gap_min']:.1f}")
        info_lines.append(f"  Max: {intermediate['h_gap_max']:.1f}")

    if 'v_gap_median' in intermediate:
        info_lines.append(f"\nVertical grid spacing (px):")
        info_lines.append(f"  Median(P50): {intermediate['v_gap_median']:.1f}")
        info_lines.append(f"  Min: {intermediate['v_gap_min']:.1f}")
        info_lines.append(f"  Max: {intermediate['v_gap_max']:.1f}")

    info_text = "\n".join(info_lines)
    outputs.append(info_text)

    stage1_data = {
        'image': result_rgb,
        'image_cv': result,
        'h_lines': intermediate.get('merged_h_lines', []),
        'v_lines': intermediate.get('merged_v_lines', []),
        'perspective_matrix': intermediate.get('perspective_matrix'),
        'grid_corners': intermediate.get('grid_corners'),
        'rotation_angle': intermediate.get('rotation_angle', 0),
        'rotation_matrix': intermediate.get('rotation_matrix'),
        'rotation_center': intermediate.get('rotation_center'),
        'original_size': intermediate.get('original_size'),
        'crop_offset': intermediate.get('crop_offset', (0, 0))
    }

    if use_extracted_chart and source_data is not None:
        stage1_data['x_tick_labels'] = source_data.get('x_tick_labels', [])
        stage1_data['y_tick_labels'] = source_data.get('y_tick_labels', [])
        stage1_data['ocr_result'] = source_data.get('ocr_result', {})
        stage1_data['original_chart_image'] = source_data.get('chart_image_cv')
        stage1_data['ai_transform_params'] = source_data.get('ai_transform_params')
        if stage0_data is not None:
            stage1_data['raw_chart_image'] = stage0_data.get('chart_image_cv')

    outputs.append(stage1_data)

    return outputs


def stage2_recognize_handler(recognizer, stage1_data, tolerance, erase_thickness,
                             color_saturation_threshold, color_value_threshold,
                             morph_connect_enabled, morph_kernel_size, morph_iterations,
                             integer_approximation_threshold):
    """Stage 2: Coordinate recognition."""
    import pandas as pd  # noqa: F811 — lazy import to avoid 52 MB IDE indexing
    lang = get_lang()

    if stage1_data is None or not isinstance(stage1_data, dict):
        return None, None, None, None, None, None, None, None, None, t("rec_err_stage1_first", lang), None, "", None

    try:
        image_cv = stage1_data['image_cv']
        h_lines = stage1_data['h_lines']
        v_lines = stage1_data['v_lines']

        x_tick_labels = stage1_data.get('x_tick_labels')
        y_tick_labels = stage1_data.get('y_tick_labels')

        rotation_matrix = stage1_data.get('rotation_matrix')
        original_size = stage1_data.get('original_size')
        crop_offset = stage1_data.get('crop_offset', (0, 0))

        if x_tick_labels is None or y_tick_labels is None:
            return None, None, None, None, None, None, None, None, None, \
                   t("rec_err_missing_labels", lang), None

        print(f"\n[Stage 2] Using Stage 0 tick info: X-axis {len(x_tick_labels)}, Y-axis {len(y_tick_labels)}")
        if rotation_matrix is not None:
            print(f"[Stage 2] Applying rotation: angle={stage1_data.get('rotation_angle', 0):.2f} deg, crop_offset: {crop_offset}")

        perspective_matrix = stage1_data.get('perspective_matrix')

        result = recognizer.process(
            image_cv, h_lines, v_lines,
            tolerance=int(tolerance),
            erase_thickness=int(erase_thickness),
            color_saturation_threshold=int(color_saturation_threshold),
            color_value_threshold=int(color_value_threshold),
            morph_connect_enabled=bool(morph_connect_enabled),
            morph_kernel_size=int(morph_kernel_size),
            morph_iterations=int(morph_iterations),
            integer_approximation_threshold=float(integer_approximation_threshold),
            x_tick_labels=x_tick_labels,
            y_tick_labels=y_tick_labels,
            perspective_matrix=perspective_matrix,
            rotation_matrix=rotation_matrix,
            original_size=original_size,
            crop_offset=crop_offset
        )

        if not result['success']:
            error_msg = result.get('error', 'Unknown error')
            return None, None, None, None, None, None, None, None, None, f"Recognition failed: {error_msg}", None, "", None

        annotated_image = result['annotated_image']
        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        intermediate_binary = result.get('intermediate_binary')
        intermediate_skeleton = result.get('intermediate_skeleton')
        intermediate_filtered = result.get('intermediate_filtered')
        intermediate_connected = result.get('intermediate_connected')

        inter_binary_rgb = cv2.cvtColor(intermediate_binary, cv2.COLOR_GRAY2RGB) if intermediate_binary is not None else None
        inter_skeleton_rgb = cv2.cvtColor(intermediate_skeleton, cv2.COLOR_GRAY2RGB) if intermediate_skeleton is not None else None
        inter_filtered_rgb = cv2.cvtColor(intermediate_filtered, cv2.COLOR_GRAY2RGB) if intermediate_filtered is not None else None
        inter_connected_rgb = cv2.cvtColor(intermediate_connected, cv2.COLOR_GRAY2RGB) if intermediate_connected is not None else None

        curve_binary = result['curve_binary']
        curve_rgb = cv2.cvtColor(curve_binary, cv2.COLOR_GRAY2RGB)

        grid_coord_vis = result.get('grid_coord_visualization')
        grid_coord_rgb = cv2.cvtColor(grid_coord_vis, cv2.COLOR_BGR2RGB) if grid_coord_vis is not None else None

        intersections_raw = result.get('intersections_raw', [])
        if len(intersections_raw) > 0:
            df_raw_data = []
            for i, pt in enumerate(intersections_raw):
                df_raw_data.append({
                    '#': i + 1,
                    'Pixel X': pt['pixel_x'],
                    'Pixel Y': pt['pixel_y'],
                    'Real X': int(round(pt['real_x'])),
                    'Real Y': int(round(pt['real_y'])),
                    'Integer Coord': f"({pt['real_x_rounded']}, {pt['real_y_rounded']})",
                    'Lines': pt['lines_count']
                })
            df_raw = pd.DataFrame(df_raw_data)
        else:
            df_raw = pd.DataFrame({'Info': ['No intersections found']})

        intersections = result['intersections']
        if len(intersections) > 0:
            df_data = []
            for i, pt in enumerate(intersections):
                df_data.append({
                    '#': i + 1,
                    'Pixel X': pt['pixel_x'],
                    'Pixel Y': pt['pixel_y'],
                    'Real X': int(round(pt['real_x'])),
                    'Real Y': int(round(pt['real_y'])),
                    'Integer Coord': f"({pt['real_x_rounded']}, {pt['real_y_rounded']})",
                    'Lines': pt['lines_count']
                })
            df = pd.DataFrame(df_data)
        else:
            df = pd.DataFrame({'Info': ['No intersections found']})

        info_lines = []
        info_lines.append(t("rec_msg_recognition_success", lang))
        info_lines.append(f"\nAxes info:")
        axes_info = result['axes_info']
        if 'x_axis_y' in axes_info:
            info_lines.append(f"  X-axis position: y = {axes_info['x_axis_y']:.1f} px")
        if 'y_axis_x' in axes_info:
            info_lines.append(f"  Y-axis position: x = {axes_info['y_axis_x']:.1f} px")
        if 'origin_x' in axes_info:
            info_lines.append(f"  Origin: ({axes_info['origin_x']:.1f}, {axes_info['origin_y']:.1f}) px")

        info_lines.append(f"\nResults:")
        info_lines.append(f"  Raw intersections: {len(intersections_raw)}")
        info_lines.append(f"  Filtered intersections: {len(intersections)}")
        info_lines.append(f"  Filter ratio: {(1 - len(intersections)/len(intersections_raw))*100:.1f}%" if len(intersections_raw) > 0 else "  Filter ratio: N/A")

        # 阶段2.5: Ruler 交叉验证 (Presenter M0 style)
        try:
            from .processor import CoordinateProcessor
            processor = CoordinateProcessor()
            ruler_enh = processor._enhance_with_ruler(
                x_tick_labels, y_tick_labels, result,
            )
            conf = ruler_enh.get('ruler_confidence', 'N/A')
            n_enh = ruler_enh.get('points_enhanced', 0)
            info_lines.append(f"\n--- Ruler Cross-Validation ---")
            info_lines.append(f"  Ruler confidence: {conf}")
            info_lines.append(f"  Points enhanced: {n_enh}")

            x_r = ruler_enh.get('x_ruler')
            y_r = ruler_enh.get('y_ruler')
            if x_r:
                info_lines.append(f"  X-axis lstsq: value = {x_r['scale']:.6f} * pixel + {x_r['offset']:.2f}  (RMSE={x_r['residual']:.4f})")
            if y_r:
                info_lines.append(f"  Y-axis lstsq: value = {y_r['scale']:.6f} * pixel + {y_r['offset']:.2f}  (RMSE={y_r['residual']:.4f})")

            cvs = ruler_enh.get('cross_validation', [])
            if cvs:
                max_dev = max(max(cv['deviation']) for cv in cvs)
                avg_dev = sum(sum(cv['deviation']) for cv in cvs) / (len(cvs) * 2)
                info_lines.append(f"  Max deviation (grid vs ruler): {max_dev:.4f}")
                info_lines.append(f"  Avg deviation (grid vs ruler): {avg_dev:.4f}")

            # 更新 result 以便传递到 Stage 3
            result['ruler_enhancement'] = ruler_enh
        except Exception as e:
            info_lines.append(f"\n--- Ruler Cross-Validation ---")
            info_lines.append(f"  Skipped: {e}")

        info_text = "\n".join(info_lines)

        # 构建 Ruler 验证 Markdown 和对比表
        ruler_md = ""
        ruler_df = None
        ruler_enh = result.get('ruler_enhancement')
        if ruler_enh:
            ruler_md_lines = []
            conf = ruler_enh.get('ruler_confidence', 'N/A')
            ruler_md_lines.append(f"### Ruler Cross-Validation (confidence: `{conf}`)")
            ruler_md_lines.append("")

            x_r = ruler_enh.get('x_ruler')
            y_r = ruler_enh.get('y_ruler')
            if x_r:
                ruler_md_lines.append(f"**X-axis regression**: `value = {x_r['scale']:.6f} × pixel + {x_r['offset']:.2f}`")
                ruler_md_lines.append(f"  - Ticks used: {x_r['num_ticks']}, RMSE: {x_r['residual']:.4f}")
            if y_r:
                ruler_md_lines.append(f"**Y-axis regression**: `value = {y_r['scale']:.6f} × pixel + {y_r['offset']:.2f}`")
                ruler_md_lines.append(f"  - Ticks used: {y_r['num_ticks']}, RMSE: {y_r['residual']:.4f}")

            cvs = ruler_enh.get('cross_validation', [])
            if cvs:
                devs = [max(cv['deviation']) for cv in cvs]
                avg_dev = sum(sum(cv['deviation']) for cv in cvs) / (len(cvs) * 2)
                ruler_md_lines.append("")
                ruler_md_lines.append(f"**Cross-validation**: {len(cvs)} points compared")
                ruler_md_lines.append(f"  - Max deviation: {max(devs):.4f}")
                ruler_md_lines.append(f"  - Avg deviation: {avg_dev:.4f}")

                # 构建对比表
                cv_data = []
                for i, cv in enumerate(cvs):
                    cv_data.append({
                        '#': i + 1,
                        'Pixel': f"({cv['pixel'][0]}, {cv['pixel'][1]})",
                        'Grid X': cv['grid'][0],
                        'Grid Y': cv['grid'][1],
                        'Ruler X': cv['ruler'][0],
                        'Ruler Y': cv['ruler'][1],
                        'Dev X': cv['deviation'][0],
                        'Dev Y': cv['deviation'][1],
                    })
                ruler_df = pd.DataFrame(cv_data)

            ruler_md = "\n".join(ruler_md_lines)

        return (inter_binary_rgb, inter_skeleton_rgb, inter_filtered_rgb, inter_connected_rgb,
                curve_rgb, grid_coord_rgb, annotated_rgb, df_raw, df, info_text, result,
                ruler_md, ruler_df)

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return None, None, None, None, None, None, None, None, None, f"Error: {str(e)}\n\n{error_detail}", None, "", None


def stage3_inverse_transform_handler(recognizer, stage1_data, stage2_result_data):
    """Stage 3: Coordinate inverse transformation and annotation."""
    import pandas as pd  # noqa: F811 — lazy import
    lang = get_lang()

    if stage1_data is None or not isinstance(stage1_data, dict):
        return None, None, None, t("rec_err_stage1_first", lang)

    if stage2_result_data is None:
        return None, None, None, t("rec_err_stage2_first", lang)

    try:
        original_chart_image = stage1_data.get('original_chart_image')
        if original_chart_image is None:
            return None, None, None, t("rec_err_missing_orig", lang)

        raw_chart_image = stage1_data.get('raw_chart_image')

        perspective_matrix = stage1_data.get('perspective_matrix')
        rotation_matrix = stage1_data.get('rotation_matrix')
        original_size = stage1_data.get('original_size')
        rotation_angle = stage1_data.get('rotation_angle', 0)
        crop_offset = stage1_data.get('crop_offset', (0, 0))
        ai_transform_params = stage1_data.get('ai_transform_params')

        if not isinstance(stage2_result_data, dict):
            return None, None, None, t("rec_err_stage2_first", lang)

        # Support both direct Stage-2 payload and nested processor payload
        # (e.g., {"result": {...}}) to avoid false "no intersections" in UI.
        stage2_payload = stage2_result_data.get('result', stage2_result_data)
        if not isinstance(stage2_payload, dict):
            stage2_payload = {}

        intersections = stage2_payload.get('intersections', []) or []
        reasoning_intersections = stage2_payload.get('reasoning_intersections', []) or []
        intersections_raw = stage2_payload.get('intersections_raw', []) or []

        # Robust fallback chain:
        # 1) visualization intersections
        # 2) reasoning intersections
        # 3) raw intersections
        if not intersections and reasoning_intersections:
            print("[Stage 3] Fallback to reasoning_intersections for inverse transform.")
            intersections = reasoning_intersections
        if not intersections and intersections_raw:
            print("[Stage 3] Fallback to intersections_raw for inverse transform.")
            intersections = intersections_raw

        if not intersections:
            print(
                "[Stage 3] No usable intersections: "
                f"filtered={len(stage2_payload.get('intersections', []) or [])}, "
                f"reasoning={len(stage2_payload.get('reasoning_intersections', []) or [])}, "
                f"raw={len(stage2_payload.get('intersections_raw', []) or [])}"
            )
            return None, None, None, t("rec_err_no_intersections_s2", lang)

        print(f"\n[Stage 3] Starting inverse transformation...")
        print(f"[Stage 3] Original image size: {original_chart_image.shape[1]}x{original_chart_image.shape[0]}")
        print(f"[Stage 3] Perspective: {'enabled' if perspective_matrix is not None else 'disabled'}")
        print(f"[Stage 3] Rotation angle: {rotation_angle:.2f} deg")
        print(f"[Stage 3] Crop offset: {crop_offset}")
        if ai_transform_params:
            print(f"[Stage 3] AI enhancement inverse transform: enabled")
        print(f"[Stage 3] Intersection count: {len(intersections)}")

        if perspective_matrix is not None or rotation_matrix is not None:
            rotated_image_shape = stage1_data['image_cv'].shape[:2]
            transformed_points = recognizer.inverse_transform_points(
                intersections, rotation_matrix, original_size, rotated_image_shape, crop_offset, perspective_matrix,
                ai_transform_params=ai_transform_params
            )
        else:
            print("[Stage 3] No inverse transform needed")
            transformed_points = intersections

        df_data = []
        for i, pt in enumerate(transformed_points):
            df_data.append({
                '#': i + 1,
                'Orig Pixel X': pt['pixel_x'],
                'Orig Pixel Y': pt['pixel_y'],
                'Corrected Pixel X': pt.get('original_pixel_x', pt['pixel_x']),
                'Corrected Pixel Y': pt.get('original_pixel_y', pt['pixel_y']),
                'Real X': int(round(pt['real_x'])),
                'Real Y': int(round(pt['real_y'])),
                'Integer Coord': f"({pt['real_x_rounded']}, {pt['real_y_rounded']})"
            })
        df_transformed = pd.DataFrame(df_data)

        def draw_annotations_on_image(image, points):
            from .core.drawing_utils import draw_intersections_on_image
            x_tick_labels = stage1_data.get('x_tick_labels', [])
            y_tick_labels = stage1_data.get('y_tick_labels', [])
            return draw_intersections_on_image(image, points, x_tick_labels, y_tick_labels)

        print("[Stage 3] Drawing annotations on AI-enhanced image...")
        annotated_ai_enhanced = draw_annotations_on_image(original_chart_image, transformed_points)
        annotated_ai_enhanced_rgb = cv2.cvtColor(annotated_ai_enhanced, cv2.COLOR_BGR2RGB)

        annotated_raw_rgb = None
        if raw_chart_image is not None:
            print("[Stage 3] Drawing annotations on original chart image...")
            annotated_raw = draw_annotations_on_image(raw_chart_image, transformed_points)
            annotated_raw_rgb = cv2.cvtColor(annotated_raw, cv2.COLOR_BGR2RGB)

        h, w = original_chart_image.shape[:2]
        info_text = f"""{t("rec_msg_stage3_complete", lang)}

Original image size: {w}x{h}
Rotation angle: {rotation_angle:.2f} deg
Intersection count: {len(transformed_points)}
"""

        return annotated_ai_enhanced_rgb, annotated_raw_rgb, df_transformed, info_text

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return None, None, None, f"Error: {str(e)}\n\n{error_detail}"
