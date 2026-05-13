"""
Gradio UI builder for the coordinate recognizer.
Can be used standalone or embedded in another Gradio app.
Supports bilingual (Chinese/English) via the centralised i18n module.
"""

import gradio as gr
from functools import partial

import sys
from pathlib import Path

_exp_root = Path(__file__).resolve().parent.parent
if str(_exp_root) not in sys.path:
    sys.path.insert(0, str(_exp_root))

from i18n import t, LANG_EN, LANG_ZH  # noqa: E402


def build_recognizer_ui(lang: str = LANG_ZH):
    """
    Build the complete recognizer UI within the current Gradio context.

    Args:
        lang: Initial language – one of LANG_EN / LANG_ZH.

    Returns:
        dict with keys:
            'markdown_components': list of (key, component) for language updates
            'interactive_components': list of (key, component, attr) for label updates
            'states': dict of gr.State components
    """
    from .core.coordinate_recognizer import CoordinateRecognizer
    from .core.chart_extractor import ChartExtractor
    from .core.image_enhancer import ImageEnhancer
    from .core.coordinate_system_preprocessor import CoordinateSystemPreprocessor
    from .ui_handlers import (
        extract_chart_handler,
        ai_process_handler,
        stage1_process_handler,
        stage2_recognize_handler,
        stage3_inverse_transform_handler,
    )

    preprocessor = CoordinateSystemPreprocessor()
    recognizer = CoordinateRecognizer()
    chart_extractor = ChartExtractor()
    image_enhancer = ImageEnhancer()

    md_components: list[tuple[str, gr.Markdown]] = []
    interactive_components: list[tuple[str, gr.components.Component, str]] = []

    def md(key):
        comp = gr.Markdown(t(key, lang))
        md_components.append((key, comp))
        return comp

    def _track(key, comp, attr="label"):
        interactive_components.append((key, comp, attr))
        return comp

    # States
    stage0_state = gr.State(value=None)
    gemini_state = gr.State(value=None)
    stage1_state = gr.State(value=None)
    stage2_state = gr.State(value=None)

    # ==================== Title ====================
    md("rec_title")
    md("rec_desc")

    # ==================== Stage 0 ====================
    md("s0_title")
    md("s0_desc")

    with gr.Row():
        with gr.Column(scale=1):
            problem_image = _track("s0_upload", gr.Image(
                label=t("s0_upload", lang), type="pil",
                sources=["upload", "webcam"],
            ))

            with gr.Accordion(t("s0_advanced", lang), open=False) as acc_s0_advanced:
                _track("s0_advanced", acc_s0_advanced)
                horizontal_tolerance = _track("s0_h_tolerance", gr.Slider(
                    minimum=0.5, maximum=3.0, value=1.5, step=0.1,
                    label=t("s0_h_tolerance", lang),
                ))
                vertical_tolerance = _track("s0_v_tolerance", gr.Slider(
                    minimum=0.5, maximum=3.0, value=1.5, step=0.1,
                    label=t("s0_v_tolerance", lang),
                ))
                interval_percentile = _track("s0_interval", gr.Slider(
                    minimum=25, maximum=75, value=50, step=5,
                    label=t("s0_interval", lang),
                ))
                gemini_enabled = _track("s0_gemini", gr.Checkbox(
                    label=t("s0_gemini", lang), value=False,
                ))

            extract_chart_btn = _track("s0_btn", gr.Button(
                t("s0_btn", lang), variant="primary", size="lg",
            ), "value")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab(t("s0_tab_result", lang)) as tab_s0_result:
                    _track("s0_tab_result", tab_s0_result)
                    stage0_chart_image = _track("s0_chart_img", gr.Image(label=t("s0_chart_img", lang)))
                    stage0_info = _track("s0_info", gr.Textbox(label=t("s0_info", lang), lines=8))

                with gr.Tab(t("s0_tab_debug", lang)) as tab_s0_debug:
                    _track("s0_tab_debug", tab_s0_debug)
                    stage0_debug_image = _track("s0_debug_vis", gr.Image(label=t("s0_debug_vis", lang)))

                with gr.Tab(t("s0_tab_ocr_info", lang)) as tab_s0_ocr_info:
                    _track("s0_tab_ocr_info", tab_s0_ocr_info)
                    stage0_debug_info = _track("s0_ocr_debug_details", gr.Textbox(
                        label=t("s0_ocr_debug_details", lang), lines=20,
                    ))

                with gr.Tab(t("s0_tab_ocr_raw", lang)) as tab_s0_ocr_raw:
                    _track("s0_tab_ocr_raw", tab_s0_ocr_raw)
                    stage0_ocr_json = _track("s0_ocr_json", gr.Code(
                        label=t("s0_ocr_json", lang), language="json", lines=20, show_label=True,
                    ))

                with gr.Tab(t("s0_tab_ocr_vis", lang)) as tab_s0_ocr_vis:
                    _track("s0_tab_ocr_vis", tab_s0_ocr_vis)
                    stage0_ocr_visual = _track("s0_ocr_vis_label", gr.Image(label=t("s0_ocr_vis_label", lang)))

                with gr.Tab(t("s0_tab_gemini", lang)) as tab_s0_gemini:
                    _track("s0_tab_gemini", tab_s0_gemini)
                    stage0_gemini_image = _track("s0_gemini_annotated", gr.Image(
                        label=t("s0_gemini_annotated", lang),
                    ))
                    stage0_gemini_comparison = gr.Markdown(t("s0_ocr_vs_gemini", lang))
                    md_components.append(("s0_ocr_vs_gemini", stage0_gemini_comparison))

    extract_chart_btn.click(
        fn=partial(extract_chart_handler, chart_extractor),
        inputs=[problem_image, horizontal_tolerance, vertical_tolerance,
                interval_percentile, gemini_enabled],
        outputs=[stage0_chart_image, stage0_state, stage0_info, stage0_debug_info,
                 stage0_debug_image, stage0_ocr_json, stage0_ocr_visual,
                 stage0_gemini_image, stage0_gemini_comparison],
    )

    gr.Markdown("---")

    # ==================== Stage 0.5 ====================
    md("s05_title")
    md("s05_desc")

    with gr.Row():
        with gr.Column(scale=1):
            enable_ai_processing = _track("s05_enable", gr.Checkbox(
                label=t("s05_enable", lang), value=False,
            ))

            model_choices = [
                ("Flux-Kontext-Pro (Azure)", "flux-kontext-pro-azure"),
                ("Flux-Kontext-Pro (Replicate)", "flux-kontext-pro"),
            ]
            model_selector = _track("s05_model", gr.Dropdown(
                choices=model_choices, value="flux-kontext-pro-azure",
                label=t("s05_model", lang),
            ))

            ai_prompt_input = _track("s05_prompt", gr.Textbox(
                label=t("s05_prompt", lang),
                placeholder=t("s05_prompt_placeholder", lang),
                lines=6,
                value=image_enhancer.get_default_prompt("flux-kontext-pro-azure"),
            ))

            ai_process_btn = _track("s05_btn", gr.Button(
                t("s05_btn", lang), variant="primary", size="lg",
            ), "value")

        with gr.Column(scale=2):
            with gr.Row():
                ai_original_image = _track("s05_original", gr.Image(label=t("s05_original", lang)))
                ai_processed_image = _track("s05_processed", gr.Image(label=t("s05_processed", lang)))
            ai_info = _track("s05_info", gr.Textbox(label=t("s05_info", lang), lines=6))

    def update_prompt_on_model_change(model_name):
        return image_enhancer.get_default_prompt(model_name)

    model_selector.change(fn=update_prompt_on_model_change,
                          inputs=[model_selector], outputs=[ai_prompt_input])

    ai_process_btn.click(
        fn=partial(ai_process_handler, image_enhancer, preprocessor),
        inputs=[stage0_state, enable_ai_processing, model_selector, ai_prompt_input],
        outputs=[ai_original_image, ai_processed_image, gemini_state, ai_info],
    )

    gr.Markdown("---")

    # ==================== Stage 1 ====================
    md("s1_title")
    md("s1_desc")

    with gr.Row():
        with gr.Column(scale=1):
            use_extracted_chart = _track("s1_use_chart", gr.Checkbox(
                label=t("s1_use_chart", lang), value=True,
            ))
            input_image = _track("s1_manual", gr.Image(
                label=t("s1_manual", lang), type="pil", sources=["upload", "webcam"],
            ))

            md("s1_steps")
            enable_curve_erase = _track("s1_curve_erase", gr.Checkbox(label=t("s1_curve_erase", lang), value=True))
            enable_dashed_connection = _track("s1_dashed", gr.Checkbox(label=t("s1_dashed", lang), value=True))
            enable_perspective = _track("s1_perspective", gr.Checkbox(label=t("s1_perspective", lang), value=True))
            enable_rotation = _track("s1_rotation", gr.Checkbox(label=t("s1_rotation", lang), value=True))

            with gr.Accordion(t("s1_curve_params", lang), open=False) as acc_s1_curve:
                _track("s1_curve_params", acc_s1_curve)
                curve_sat_thresh = _track("s2_sat_thresh", gr.Slider(
                    50, 200, value=90, step=5, label=t("s2_sat_thresh", lang),
                ))
                curve_val_thresh = _track("s2_val_thresh", gr.Slider(
                    50, 200, value=60, step=5, label=t("s2_val_thresh", lang),
                ))
                curve_feather = _track("s1_feather_size", gr.Slider(
                    0, 10, value=3, step=1, label=t("s1_feather_size", lang),
                ))

            with gr.Accordion(t("s1_dashed_params", lang), open=False) as acc_s1_dashed:
                _track("s1_dashed_params", acc_s1_dashed)
                dashed_gap_thresh = _track("s1_gap_threshold", gr.Slider(
                    1, 100, value=10, step=1, label=t("s1_gap_threshold", lang),
                ))
                dashed_collinear_thresh = _track("s1_collinear_threshold", gr.Slider(
                    1, 30, value=3, step=1, label=t("s1_collinear_threshold", lang),
                ))

            with gr.Accordion(t("s1_rot_params", lang), open=False) as acc_s1_rot:
                _track("s1_rot_params", acc_s1_rot)
                rot_hough_thresh = _track("s1_hough_threshold", gr.Slider(
                    10, 200, value=60, step=10, label=t("s1_hough_threshold", lang),
                ))
                rot_min_line_len = _track("s1_min_line_length", gr.Slider(
                    20, 200, value=60, step=10, label=t("s1_min_line_length", lang),
                ))
                rot_max_gap = _track("s1_max_line_gap", gr.Slider(
                    5, 50, value=10, step=5, label=t("s1_max_line_gap", lang),
                ))
                rot_angle_thresh = _track("s1_angle_threshold", gr.Slider(
                    5, 20, value=5, step=1, label=t("s1_angle_threshold", lang),
                ))

            with gr.Accordion(t("s1_grid_params", lang), open=False) as acc_s1_grid:
                _track("s1_grid_params", acc_s1_grid)
                grid_hough_thresh = _track("s1_hough_threshold", gr.Slider(
                    10, 200, value=65, step=5, label=t("s1_hough_threshold", lang),
                ))
                grid_min_line_len = _track("s1_min_line_length", gr.Slider(
                    10, 200, value=45, step=5, label=t("s1_min_line_length", lang),
                ))
                grid_max_gap = _track("s1_max_line_gap", gr.Slider(
                    5, 50, value=10, step=5, label=t("s1_max_line_gap", lang),
                ))
                grid_angle_thresh = _track("s1_angle_threshold", gr.Slider(
                    5, 20, value=5, step=1, label=t("s1_angle_threshold", lang),
                ))

            with gr.Accordion(t("s1_merge_params", lang), open=False) as acc_s1_merge:
                _track("s1_merge_params", acc_s1_merge)
                merge_dist_thresh = _track("s1_merge_dist", gr.Slider(
                    5, 100, value=20, step=5, label=t("s1_merge_dist", lang),
                ))

            process_btn = _track("s1_btn", gr.Button(
                t("s1_btn", lang), variant="primary", size="lg",
            ), "value")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab(t("s1_tab_final", lang)) as tab_s1_final:
                    _track("s1_tab_final", tab_s1_final)
                    output_final = _track("s1_processed_img", gr.Image(label=t("s1_processed_img", lang)))
                with gr.Tab(t("s1_tab_edges", lang)) as tab_s1_edges:
                    _track("s1_tab_edges", tab_s1_edges)
                    output_edges = _track("s1_edge_detect", gr.Image(label=t("s1_edge_detect", lang)))
                with gr.Tab(t("s1_tab_lines", lang)) as tab_s1_lines:
                    _track("s1_tab_lines", tab_s1_lines)
                    output_lines = _track("s1_detected_lines", gr.Image(label=t("s1_detected_lines", lang)))
                with gr.Tab(t("s1_tab_dashed", lang)) as tab_s1_dashed:
                    _track("s1_tab_dashed", tab_s1_dashed)
                    output_dashed = _track("s1_dashed_conn", gr.Image(label=t("s1_dashed_conn", lang)))
                with gr.Tab(t("s1_tab_merged", lang)) as tab_s1_merged:
                    _track("s1_tab_merged", tab_s1_merged)
                    output_merged = _track("s1_merged_grid", gr.Image(label=t("s1_merged_grid", lang)))

            s1_info_text = _track("s1_info", gr.Textbox(label=t("s1_info", lang), lines=10))

    process_btn.click(
        fn=partial(stage1_process_handler, preprocessor),
        inputs=[
            use_extracted_chart, stage0_state, gemini_state, input_image,
            enable_curve_erase, enable_dashed_connection, enable_perspective, enable_rotation,
            curve_sat_thresh, curve_val_thresh, curve_feather,
            dashed_gap_thresh, dashed_collinear_thresh,
            rot_hough_thresh, rot_min_line_len, rot_max_gap, rot_angle_thresh,
            grid_hough_thresh, grid_min_line_len, grid_max_gap, grid_angle_thresh,
            merge_dist_thresh,
        ],
        outputs=[
            output_final, output_edges, output_lines, output_dashed, output_merged,
            s1_info_text, stage1_state,
        ],
    )

    gr.Markdown("---")

    # ==================== Stage 2 ====================
    md("s2_title")
    md("s2_desc")

    with gr.Row():
        with gr.Column(scale=1):
            md("s2_params")

            intersection_tolerance = _track("s2_tolerance", gr.Slider(
                1, 20, value=3, step=1, label=t("s2_tolerance", lang),
            ))
            erase_thickness = _track("s2_erase", gr.Slider(
                1, 15, value=5, step=1, label=t("s2_erase", lang),
            ))

            with gr.Accordion(t("s2_curve_extract", lang), open=True) as acc_s2_curve:
                _track("s2_curve_extract", acc_s2_curve)
                color_saturation_threshold = _track("s2_sat_thresh", gr.Slider(
                    50, 200, value=90, step=5, label=t("s2_sat_thresh", lang),
                ))
                color_value_threshold = _track("s2_val_thresh", gr.Slider(
                    50, 200, value=60, step=5, label=t("s2_val_thresh", lang),
                ))
                morph_connect_enabled = _track("s2_morph_enable", gr.Checkbox(
                    value=True, label=t("s2_morph_enable", lang),
                ))
                morph_kernel_size = _track("s2_morph_kernel", gr.Slider(
                    3, 21, value=7, step=2, label=t("s2_morph_kernel", lang),
                ))
                morph_iterations = _track("s2_morph_iter", gr.Slider(
                    1, 5, value=2, step=1, label=t("s2_morph_iter", lang),
                ))

            with gr.Accordion(t("s2_filter_params", lang), open=True) as acc_s2_filter:
                _track("s2_filter_params", acc_s2_filter)
                integer_approx_threshold = _track("s2_integer_thresh", gr.Slider(
                    0.0, 0.5, value=0.1, step=0.05, label=t("s2_integer_thresh", lang),
                ))

            recognize_btn = _track("s2_btn", gr.Button(
                t("s2_btn", lang), variant="primary", size="lg",
            ), "value")

        with gr.Column(scale=2):
            stage2_info = _track("s2_info", gr.Textbox(label=t("s2_info", lang), lines=8))

            with gr.Tabs():
                with gr.Tab(t("s2_tab_intermediate", lang)) as tab_s2_inter:
                    _track("s2_tab_intermediate", tab_s2_inter)
                    with gr.Row():
                        with gr.Column():
                            intermediate_binary = _track("s2_binarized", gr.Image(label=t("s2_binarized", lang)))
                            intermediate_skeleton = _track("s2_skeleton", gr.Image(label=t("s2_skeleton", lang)))
                        with gr.Column():
                            intermediate_filtered = _track("s2_filtered", gr.Image(label=t("s2_filtered", lang)))
                            intermediate_connected = _track("s2_connected", gr.Image(label=t("s2_connected", lang)))

                with gr.Tab(t("s2_tab_curve", lang)) as tab_s2_curve:
                    _track("s2_tab_curve", tab_s2_curve)
                    stage2_curve = _track("s2_extracted_curve", gr.Image(label=t("s2_extracted_curve", lang)))

                with gr.Tab(t("s2_tab_grid_coords", lang)) as tab_s2_grid:
                    _track("s2_tab_grid_coords", tab_s2_grid)
                    stage2_grid_coords = _track("s2_grid_coord_map", gr.Image(label=t("s2_grid_coord_map", lang)))

                with gr.Tab(t("s2_tab_annotated", lang)) as tab_s2_annotated:
                    _track("s2_tab_annotated", tab_s2_annotated)
                    stage2_annotated = _track("s2_annotated_img", gr.Image(label=t("s2_annotated_img", lang)))

                with gr.Tab(t("s2_tab_raw_table", lang)) as tab_s2_raw:
                    _track("s2_tab_raw_table", tab_s2_raw)
                    stage2_table_raw = _track("s2_raw_intersections", gr.Dataframe(
                        label=t("s2_raw_intersections", lang),
                        headers=["#", "Pixel X", "Pixel Y", "Real X", "Real Y", "Integer Coord", "Lines"],
                    ))

                with gr.Tab(t("s2_tab_filtered_table", lang)) as tab_s2_filtered:
                    _track("s2_tab_filtered_table", tab_s2_filtered)
                    stage2_table = _track("s2_filtered_intersections", gr.Dataframe(
                        label=t("s2_filtered_intersections", lang),
                        headers=["#", "Pixel X", "Pixel Y", "Real X", "Real Y", "Integer Coord", "Lines"],
                    ))

                with gr.Tab(t("s2_tab_ruler", lang)) as tab_s2_ruler:
                    _track("s2_tab_ruler", tab_s2_ruler)
                    stage2_ruler_info = _track("s2_ruler_info", gr.Markdown(
                        label=t("s2_ruler_info", lang),
                    ))
                    stage2_ruler_table = _track("s2_ruler_table", gr.Dataframe(
                        label=t("s2_ruler_table", lang),
                        headers=["#", "Pixel", "Grid X", "Grid Y", "Ruler X", "Ruler Y", "Dev X", "Dev Y"],
                    ))

    recognize_btn.click(
        fn=partial(stage2_recognize_handler, recognizer),
        inputs=[stage1_state, intersection_tolerance, erase_thickness,
                color_saturation_threshold, color_value_threshold,
                morph_connect_enabled, morph_kernel_size, morph_iterations,
                integer_approx_threshold],
        outputs=[intermediate_binary, intermediate_skeleton, intermediate_filtered, intermediate_connected,
                 stage2_curve, stage2_grid_coords, stage2_annotated, stage2_table_raw, stage2_table,
                 stage2_info, stage2_state, stage2_ruler_info, stage2_ruler_table],
    )

    gr.Markdown("---")

    # ==================== Stage 3 ====================
    md("s3_title")
    md("s3_desc")

    with gr.Row():
        with gr.Column(scale=1):
            stage3_process_btn = _track("s3_btn", gr.Button(
                t("s3_btn", lang), variant="primary", size="lg",
            ), "value")

        with gr.Column(scale=2):
            stage3_info = _track("s3_info", gr.Textbox(label=t("s3_info", lang), lines=6))

            with gr.Tabs():
                with gr.Tab(t("s3_tab_ai", lang)) as tab_s3_ai:
                    _track("s3_tab_ai", tab_s3_ai)
                    stage3_annotated_ai = _track("s3_ai_annotation", gr.Image(
                        label=t("s3_ai_annotation", lang),
                    ))

                with gr.Tab(t("s3_tab_raw", lang)) as tab_s3_raw:
                    _track("s3_tab_raw", tab_s3_raw)
                    stage3_annotated_raw = _track("s3_orig_annotation", gr.Image(
                        label=t("s3_orig_annotation", lang),
                    ))

                with gr.Tab(t("s3_tab_table", lang)) as tab_s3_table:
                    _track("s3_tab_table", tab_s3_table)
                    stage3_table = _track("s3_coord_comparison", gr.Dataframe(
                        label=t("s3_coord_comparison", lang),
                        headers=["#", "Orig Pixel X", "Orig Pixel Y",
                                 "Corrected Pixel X", "Corrected Pixel Y",
                                 "Real X", "Real Y", "Integer Coord"],
                    ))

    stage3_process_btn.click(
        fn=partial(stage3_inverse_transform_handler, recognizer),
        inputs=[stage1_state, stage2_state],
        outputs=[stage3_annotated_ai, stage3_annotated_raw, stage3_table, stage3_info],
    )

    return {
        "markdown_components": md_components,
        "interactive_components": interactive_components,
        "states": {
            "stage0": stage0_state,
            "gemini": gemini_state,
            "stage1": stage1_state,
            "stage2": stage2_state,
        },
        "gemini_enabled": gemini_enabled,
    }


def update_recognizer_language(md_components, interactive_components, lang):
    """
    Return updated values for all tracked recognizer components.

    Args:
        md_components: list of (key, component) tuples
        interactive_components: list of (key, component, attr) tuples
        lang: LANG_EN or LANG_ZH

    Returns:
        list of gr.update / text values (markdown first, then interactive)
    """
    updates: list = []
    for key, _ in md_components:
        updates.append(t(key, lang))
    for key, _, attr in interactive_components:
        if attr == "value":
            updates.append(gr.update(value=t(key, lang)))
        else:
            updates.append(gr.update(label=t(key, lang)))
    return updates
