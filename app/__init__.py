#!/usr/bin/env python3
"""
Experiment Evaluation UI - Gradio interface.

This is a thin UI shell that imports all backend functions from:
  - app.helpers: constants, config, dataset, LLM adapter
  - app.generation: image generation and preview
  - app.reasoning: math reasoning pipeline
"""

import os
import warnings
from pathlib import Path
from typing import Any

# Suppress Gradio SVG watermark warning
warnings.filterwarnings("ignore", message=".*Watermarking for SVG.*")

import gradio as gr

# ---------------------------------------------------------------------------
# Imports from extracted backend modules
# ---------------------------------------------------------------------------
from app.helpers import (
    _root,
    custom_css,
    NAVY_BLUE, DARK_NAVY, ACCENT_BLUE, TEXT_COLOR,
    list_dataset_images, load_image_and_gt,
    PROVIDER_MAP, PROVIDER_NAMES,
)
from app.generation import (
    generate_images, clear_dataset_ui,
    _build_type_cap_md,
    preview_image, refresh_dataset,
    FUNCTION_TYPE_LABELS, DEFAULT_FUNC_TYPE_LABELS,
    profile_choices, DEFAULT_PROFILE_ID,
    generation_mode_choices, benchmark_dimensions_md,
    sample_plan_choices, DEFAULT_SAMPLE_PLAN_ID, sample_plan_markdown,
)
from app.reasoning import run_reasoning_all_conditions
from app.facts_eval import run_facts_diagnosis
from app.reasoning_cache import clear_image_reasoning_cache, clear_all_reasoning_cache
from app.reasoning_pipeline import CONDITION_ORDER, resolve_image_path as _resolve_image_path
from app.connection_test import (
    build_provider_overview, run_single_connection_test,
)
from presenter import (
    list_real_chart_samples,
    build_sample_overview as build_presenter_sample_overview,
    get_sample_image_path as get_presenter_sample_image_path,
    run_presenter_showcase,
)

from i18n import t, LANG_EN, LANG_ZH, set_lang
from generator.rendering.curves import ALL_FUNCTION_TYPES, FUNCTION_TYPE_DESCRIPTIONS
from recognizer.ui_builder import build_recognizer_ui, update_recognizer_language


# ==================== Gradio App ====================

def create_app():
    """Create and return the Gradio app."""

    with gr.Blocks(
        title="Experiment Evaluation System",
    ) as app:

        # Language selector at the top
        with gr.Row():
            lang_selector = gr.Radio(
                [LANG_EN, LANG_ZH],
                value=LANG_ZH,
                label="Language / 语言",
                scale=1,
            )

        _empty_key = gr.State("")

        # Title (will be updated by language)
        _desc = t('app_desc', LANG_ZH)
        title_md = gr.Markdown(f"# {t('app_title', LANG_ZH)}\n{_desc}" if _desc else f"# {t('app_title', LANG_ZH)}")

        def _preview_image_path(image_name):
            if not image_name:
                return None
            p = _resolve_image_path(image_name)
            return str(p) if p else None

        with gr.Tabs() as tabs:
            # ===== Tab 1: GeoSym Full Function Reasoning =====
            with gr.Tab(t("tab_reasoning", LANG_ZH)) as tab_reasoning:
                reason_title_md = gr.Markdown(t("reason_title", LANG_ZH))

                with gr.Accordion(t("reason_conditions_title", LANG_ZH), open=False):
                    reason_conditions_md = gr.Markdown(t("reason_conditions_desc", LANG_ZH))

                _default_provider = PROVIDER_NAMES[0] if PROVIDER_NAMES else ""
                _default_models = PROVIDER_MAP.get(_default_provider, [])
                _default_model = _default_models[0] if _default_models else ""

                with gr.Row():
                    reason_provider = gr.Radio(
                        PROVIDER_NAMES,
                        value=_default_provider,
                        label=t("reason_provider", LANG_ZH),
                        scale=2,
                    )
                    reason_model_dd = gr.Dropdown(
                        choices=_default_models,
                        value=_default_model,
                        label=t("reason_model", LANG_ZH),
                        interactive=True,
                        scale=2,
                    )

                reason_model = gr.State(_default_model)

                def _on_reason_provider_change(provider):
                    models = PROVIDER_MAP.get(provider, [])
                    return gr.update(choices=models, value=models[0] if models else None)

                def _sync_reason_model(provider, model):
                    return model or ""

                reason_provider.change(_on_reason_provider_change, inputs=[reason_provider], outputs=[reason_model_dd])
                reason_provider.change(_sync_reason_model, inputs=[reason_provider, reason_model_dd], outputs=[reason_model])
                reason_model_dd.change(_sync_reason_model, inputs=[reason_provider, reason_model_dd], outputs=[reason_model])

                with gr.Row():
                    judge_provider = gr.Radio(
                        PROVIDER_NAMES,
                        value=_default_provider,
                        label=t("judge_provider", LANG_ZH),
                        scale=2,
                    )
                    judge_model_dd = gr.Dropdown(
                        choices=_default_models,
                        value=_default_model,
                        label=t("judge_model", LANG_ZH),
                        interactive=True,
                        scale=2,
                    )

                judge_model = gr.State(_default_model)

                judge_provider.change(_on_reason_provider_change, inputs=[judge_provider], outputs=[judge_model_dd])
                judge_provider.change(_sync_reason_model, inputs=[judge_provider, judge_model_dd], outputs=[judge_model])
                judge_model_dd.change(_sync_reason_model, inputs=[judge_provider, judge_model_dd], outputs=[judge_model])

                with gr.Row():
                    reason_single_image = gr.Dropdown(
                        choices=list_dataset_images(),
                        allow_custom_value=True,
                        label=t("reason_select_image", LANG_ZH),
                        scale=3,
                    )
                    reason_all_btn = gr.Button(t("reason_btn_run_all", LANG_ZH), variant="primary", scale=1)
                    reason_retry_btn = gr.Button(t("reason_btn_retry", LANG_ZH), variant="stop", scale=1)
                    reason_rerun_btn = gr.Button(t("reason_btn_rerun_all", LANG_ZH), variant="secondary", scale=1)
                    reason_clear_btn = gr.Button(t("reason_btn_clear_cache", LANG_ZH), variant="huggingface", scale=1)
                    reason_clear_all_btn = gr.Button(t("reason_btn_clear_all_cache", LANG_ZH), variant="stop", scale=1)

                with gr.Row():
                    reason_rerun_condition_dd = gr.Dropdown(
                        choices=list(CONDITION_ORDER),
                        value="E0",
                        label=t("reason_select_rerun_condition", LANG_ZH),
                        scale=2,
                    )
                    reason_rerun_condition_btn = gr.Button(
                        t("reason_btn_rerun_selected", LANG_ZH), variant="secondary", scale=1,
                    )
                with gr.Row():
                    with gr.Column(scale=1):
                        reason_preview_img = gr.Image(
                            label=t("reason_preview", LANG_ZH),
                            type="filepath",
                            interactive=False,
                        )
                    with gr.Column(scale=2):
                        reason_gt_md = gr.Markdown(label=t("reason_ground_truth", LANG_ZH))

                reason_single_image.change(
                    _preview_image_path,
                    inputs=[reason_single_image],
                    outputs=[reason_preview_img],
                )

                reason_summary_md = gr.Markdown(label=t("reason_summary_table", LANG_ZH))

                with gr.Tabs():
                    with gr.Tab(t("reason_cond_e0", LANG_ZH)) as tab_cond_e0:
                        reason_cond_e0_md = gr.Markdown()
                    with gr.Tab(t("reason_cond_e1", LANG_ZH)) as tab_cond_e1:
                        reason_cond_e1_md = gr.Markdown()
                    with gr.Tab(t("reason_cond_e2", LANG_ZH)) as tab_cond_e2:
                        reason_cond_e2_md = gr.Markdown()
                    with gr.Tab(t("reason_cond_e3", LANG_ZH)) as tab_cond_e3:
                        reason_cond_e3_md = gr.Markdown()
                    with gr.Tab(t("reason_cond_e4", LANG_ZH)) as tab_cond_e4:
                        reason_cond_e4_md = gr.Markdown()
                    with gr.Tab(t("reason_cond_e5", LANG_ZH)) as tab_cond_e5:
                        reason_cond_e5_md = gr.Markdown()
                    with gr.Tab(t("reason_cond_e5_raw", LANG_ZH)) as tab_cond_e5_raw:
                        reason_cond_e5_raw_md = gr.Markdown()

            # ===== Tab 2: GeoSym Function-Specialized M0 Demo =====
            with gr.Tab(t("tab_recognizer", LANG_ZH)) as tab_recognizer:
                rec_tab_title_md = gr.Markdown(t("rec_tab_title", LANG_ZH))
                recognizer_result = build_recognizer_ui(lang=LANG_ZH)
                _rec_md_components = recognizer_result['markdown_components']
                _rec_interactive_components = recognizer_result['interactive_components']
                gemini_enabled_checkbox = recognizer_result['gemini_enabled']

            # ===== Tab 3: GeoSym Function-Specialized M0 Test =====
            with gr.Tab(t("tab_facts_accuracy", LANG_ZH)) as tab_facts_accuracy:
                facts_title_md = gr.Markdown(t("facts_tab_title", LANG_ZH))
                facts_desc_md = gr.Markdown(t("facts_tab_desc", LANG_ZH))

                with gr.Row():
                    facts_image_dd = gr.Dropdown(
                        choices=list_dataset_images(),
                        allow_custom_value=True,
                        label=t("facts_select_image", LANG_ZH),
                        scale=3,
                    )
                    facts_run_btn = gr.Button(t("facts_run_btn", LANG_ZH), variant="primary", scale=1)

                with gr.Row():
                    facts_preview_img = gr.Image(
                        label=t("facts_preview", LANG_ZH),
                        type="filepath",
                        interactive=False,
                    )
                    with gr.Column(scale=2):
                        facts_summary_md = gr.Markdown()
                        facts_detail_md = gr.Markdown()

                facts_raw_md = gr.Markdown()
                facts_annotated_img = gr.Image(
                    label=t("s3_orig_annotation", LANG_ZH),
                    interactive=False,
                )

                facts_image_dd.change(
                    _preview_image_path,
                    inputs=[facts_image_dd],
                    outputs=[facts_preview_img],
                )

                facts_run_btn.click(
                    run_facts_diagnosis,
                    inputs=[facts_image_dd, gemini_enabled_checkbox, lang_selector, reason_model],
                    outputs=[facts_summary_md, facts_detail_md, facts_raw_md, facts_annotated_img],
                )

            _all_cond_outputs = [
                reason_gt_md, reason_summary_md,
                reason_cond_e0_md, reason_cond_e1_md, reason_cond_e2_md,
                reason_cond_e3_md, reason_cond_e4_md, reason_cond_e5_md,
                reason_cond_e5_raw_md,
            ]

            _reason_inputs = [
                reason_single_image, reason_model, _empty_key,
                judge_model, _empty_key, lang_selector,
                gemini_enabled_checkbox,
            ]

            def _run_all_wrapper(img, model, key, jmodel, jkey, lang, gemini):
                yield from run_reasoning_all_conditions(
                    img, model, key, jmodel, jkey, lang, gemini,
                    force_rerun=False)

            def _run_retry_wrapper(img, model, key, jmodel, jkey, lang, gemini):
                yield from run_reasoning_all_conditions(
                    img, model, key, jmodel, jkey, lang, gemini,
                    force_rerun=False)

            reason_all_btn.click(
                _run_all_wrapper,
                inputs=_reason_inputs,
                outputs=_all_cond_outputs,
            )

            reason_retry_btn.click(
                _run_retry_wrapper,
                inputs=_reason_inputs,
                outputs=_all_cond_outputs,
            )

            def _run_rerun_wrapper(img, model, key, jmodel, jkey, lang, gemini):
                yield from run_reasoning_all_conditions(
                    img, model, key, jmodel, jkey, lang, gemini,
                    force_rerun=True)

            def _run_selected_condition_wrapper(img, model, key, jmodel, jkey, lang, gemini, cond_key):
                yield from run_reasoning_all_conditions(
                    img, model, key, jmodel, jkey, lang, gemini,
                    force_rerun=True,
                    target_conditions=[cond_key] if cond_key else None,
                )

            reason_rerun_btn.click(
                _run_rerun_wrapper,
                inputs=_reason_inputs,
                outputs=_all_cond_outputs,
            )

            reason_rerun_condition_btn.click(
                _run_selected_condition_wrapper,
                inputs=_reason_inputs + [reason_rerun_condition_dd],
                outputs=_all_cond_outputs,
            )

            def _clear_current_image_cache(img, lang):
                if not img:
                    return gr.update(value=t("reason_clear_no_image", lang))
                n = clear_image_reasoning_cache(img)
                if n > 0:
                    return gr.update(value=t("reason_clear_confirm", lang).format(files=n))
                return gr.update(value=t("reason_clear_empty", lang))

            reason_clear_btn.click(
                _clear_current_image_cache,
                inputs=[reason_single_image, lang_selector],
                outputs=[reason_summary_md],
            )

            def _clear_all_reasoning_cache(lang):
                n = clear_all_reasoning_cache()
                if n > 0:
                    return gr.update(value=t("reason_clear_all_confirm", lang).format(count=n))
                return gr.update(value=t("reason_clear_all_empty", lang))

            reason_clear_all_btn.click(
                _clear_all_reasoning_cache,
                inputs=[lang_selector],
                outputs=[reason_summary_md],
            )

            # ===== Tab 4: Image Generation + Dataset Preview =====
            with gr.Tab(t("tab_generation", LANG_ZH)) as tab_gen:
                gen_title_md = gr.Markdown(t("gen_title", LANG_ZH))

                with gr.Row():
                    with gr.Column(scale=1):
                        gen_basic_md = gr.Markdown(t("gen_basic", LANG_ZH))
                        gen_count = gr.Slider(1, 1000, value=10, step=1, label=t("gen_count", LANG_ZH))
                        gen_seed = gr.Number(value=42, label=t("gen_seed", LANG_ZH), precision=0)
                        gen_mode = gr.Radio(
                            generation_mode_choices(LANG_ZH),
                            value=t("gen_mode_selected", LANG_ZH),
                            label=t("gen_mode", LANG_ZH),
                        )
                        gen_sample_plan = gr.Radio(
                            choices=sample_plan_choices(LANG_ZH),
                            value=DEFAULT_SAMPLE_PLAN_ID,
                            label=t("gen_sample_plan", LANG_ZH),
                        )
                        gen_sample_plan_md = gr.Markdown(
                            sample_plan_markdown(
                                LANG_ZH,
                                DEFAULT_SAMPLE_PLAN_ID,
                                10,
                                DEFAULT_FUNC_TYPE_LABELS,
                                t("gen_mode_selected", LANG_ZH),
                            )
                        )

                    with gr.Column(scale=2):
                        gen_bench_md = gr.Markdown(benchmark_dimensions_md(LANG_ZH))
                        with gr.Row():
                            gen_func_types = gr.CheckboxGroup(
                                FUNCTION_TYPE_LABELS,
                                value=DEFAULT_FUNC_TYPE_LABELS,
                                label=t("gen_func_types", LANG_ZH),
                            )
                            gen_profile = gr.Radio(
                                choices=profile_choices(LANG_ZH),
                                value=DEFAULT_PROFILE_ID,
                                label=t("gen_profile", LANG_ZH),
                                visible=False,
                            )
                        gen_func_desc_md = gr.Markdown(t("gen_func_desc", LANG_ZH))
                        gen_coord_mode = gr.Radio(
                            [t("gen_coord_mode_adaptive", LANG_ZH),
                             t("gen_coord_mode_static", LANG_ZH),
                             t("gen_coord_mode_manual", LANG_ZH)],
                            value=t("gen_coord_mode_adaptive", LANG_ZH),
                            label=t("gen_coord_mode", LANG_ZH),
                        )

                with gr.Accordion(t("gen_advanced", LANG_ZH), open=False) as acc_gen_advanced:
                    with gr.Row():
                        with gr.Column(scale=1):
                            gen_page_md = gr.Markdown(t("gen_page_dim", LANG_ZH))
                            page_width = gr.Slider(800, 2560, value=1440, step=64, label=t("gen_page_width", LANG_ZH))
                            page_height = gr.Slider(600, 1920, value=1440, step=64, label=t("gen_page_height", LANG_ZH))

                        with gr.Column(scale=1):
                            gen_coord_md = gr.Markdown(t("gen_coord_range", LANG_ZH))
                            x_min = gr.Slider(-20, 0, value=-5, step=1, label=t("gen_x_min", LANG_ZH))
                            x_max = gr.Slider(0, 20, value=5, step=1, label=t("gen_x_max", LANG_ZH))
                            y_min = gr.Slider(-20, 0, value=-5, step=1, label=t("gen_y_min", LANG_ZH))
                            y_max = gr.Slider(0, 20, value=5, step=1, label=t("gen_y_max", LANG_ZH))
                            min_intersections = gr.Slider(0, 15, value=6, step=1, label=t("gen_min_intersections", LANG_ZH))

                        with gr.Column(scale=1):
                            type_cap_title = gr.Markdown(f"**{t('gen_type_cap_title', LANG_ZH)}**")
                            type_cap_md = gr.Markdown(_build_type_cap_md(LANG_ZH))

                        with gr.Column(scale=1):
                            gen_curves_md = gr.Markdown(t("gen_curves_noise", LANG_ZH))
                            noise_enabled = gr.Checkbox(value=True, label=t("gen_noise_enabled", LANG_ZH))

                with gr.Row():
                    gen_btn = gr.Button(t("gen_btn", LANG_ZH), variant="primary", scale=3)
                    gen_clear_btn = gr.Button(t("gen_clear_btn", LANG_ZH), variant="stop", scale=1)

                with gr.Row():
                    gen_status = gr.Textbox(label=t("gen_status", LANG_ZH), lines=2)
                    gen_stats = gr.Textbox(label=t("gen_stats", LANG_ZH), lines=2)


                _gen_event = gen_btn.click(
                    generate_images,
                    inputs=[gen_count, gen_seed, gen_mode,
                            gen_sample_plan,
                            gen_func_types, gen_profile,
                            gen_coord_mode,
                            page_width, page_height,
                            x_min, x_max, y_min, y_max,
                            noise_enabled, min_intersections,
                            lang_selector],
                    outputs=[gen_status, gen_stats],
                )

                def _refresh_sample_plan(lang, plan_id, count, func_types, mode):
                    return sample_plan_markdown(lang, plan_id, int(count), func_types, mode)

                for trigger in (gen_count, gen_mode, gen_sample_plan, gen_func_types):
                    trigger.change(
                        _refresh_sample_plan,
                        inputs=[lang_selector, gen_sample_plan, gen_count, gen_func_types, gen_mode],
                        outputs=[gen_sample_plan_md],
                    )

                lang_selector.change(
                    _refresh_sample_plan,
                    inputs=[lang_selector, gen_sample_plan, gen_count, gen_func_types, gen_mode],
                    outputs=[gen_sample_plan_md],
                )

                gen_clear_btn.click(
                    clear_dataset_ui,
                    inputs=[lang_selector],
                    outputs=[gen_status, gen_stats],
                )

                # --- Dataset Preview (merged) ---
                gr.Markdown("---")
                preview_title_md = gr.Markdown(t("preview_title", LANG_ZH))

                with gr.Row():
                    dataset_dropdown = gr.Dropdown(
                        choices=list_dataset_images(),
                        allow_custom_value=True,
                        label=t("preview_select", LANG_ZH),
                        scale=3,
                    )
                    refresh_btn = gr.Button(t("preview_refresh", LANG_ZH), scale=1)

                _latex_delims = [
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ]

                with gr.Row(equal_height=False):
                    preview_img = gr.Image(
                        label=t("preview_img", LANG_ZH),
                        type="numpy",
                        height=520,
                        scale=1,
                    )
                    with gr.Column(scale=1):
                        with gr.Accordion(t("preview_acc_expr", LANG_ZH), open=True) as pv_acc_expr:
                            pv_expr_md = gr.Markdown(
                                value="", latex_delimiters=_latex_delims,
                            )
                        with gr.Accordion(t("preview_acc_props", LANG_ZH), open=True) as pv_acc_props:
                            pv_props_md = gr.Markdown(
                                value="", latex_delimiters=_latex_delims,
                            )
                        with gr.Accordion(t("preview_acc_extended", LANG_ZH), open=False) as pv_acc_extended:
                            pv_extended_md = gr.Markdown(
                                value="", latex_delimiters=_latex_delims,
                            )
                        with gr.Accordion(t("preview_acc_ints", LANG_ZH), open=False) as pv_acc_ints:
                            pv_ints_md = gr.Markdown(
                                value="", latex_delimiters=_latex_delims,
                            )
                        with gr.Accordion(t("preview_acc_tasks", LANG_ZH), open=False) as pv_acc_tasks:
                            pv_tasks_md = gr.Markdown(
                                value="", latex_delimiters=_latex_delims,
                            )
                        with gr.Accordion(t("preview_acc_meta", LANG_ZH), open=False) as pv_acc_meta:
                            pv_meta_md = gr.Markdown(value="")

                dataset_dropdown.change(
                    preview_image,
                    inputs=[dataset_dropdown, lang_selector],
                    outputs=[preview_img, pv_expr_md, pv_props_md,
                             pv_extended_md, pv_ints_md, pv_tasks_md, pv_meta_md],
                )

            # ===== Tab 5: Connection Test =====
            with gr.Tab(t("tab_conn_test", LANG_ZH)) as tab_conn_test:
                conn_title_md = gr.Markdown(t("conn_test_title", LANG_ZH))

                conn_overview_md_title = gr.Markdown(t("conn_overview_title", LANG_ZH))
                conn_overview_md = gr.Markdown(
                    value=build_provider_overview(LANG_ZH),
                )

                _conn_first_provider = PROVIDER_NAMES[0] if PROVIDER_NAMES else ""
                _conn_first_models = PROVIDER_MAP.get(_conn_first_provider, [])
                with gr.Row():
                    conn_provider = gr.Radio(
                        PROVIDER_NAMES,
                        value=_conn_first_provider,
                        label=t("conn_provider", LANG_ZH),
                        scale=2,
                    )
                    conn_model_dd = gr.Dropdown(
                        choices=_conn_first_models,
                        value=_conn_first_models[0] if _conn_first_models else None,
                        label=t("conn_model", LANG_ZH),
                        interactive=True,
                        scale=2,
                    )

                def _on_conn_provider_change(provider):
                    models = PROVIDER_MAP.get(provider, [])
                    return gr.update(choices=models, value=models[0] if models else None)

                conn_provider.change(
                    _on_conn_provider_change,
                    inputs=[conn_provider],
                    outputs=[conn_model_dd],
                )

                with gr.Row():
                    conn_test_btn = gr.Button(
                        t("conn_test_btn", LANG_ZH), variant="primary", scale=1,
                    )
                conn_result_md = gr.Markdown(
                    value="", label=t("conn_result_label", LANG_ZH),
                )

                conn_test_btn.click(
                    run_single_connection_test,
                    inputs=[conn_model_dd, lang_selector],
                    outputs=[conn_result_md],
                )

            # ===== Tab 6: Real Chart Presenter =====
            with gr.Tab(t("tab_presenter", LANG_ZH)) as tab_presenter:
                presenter_title_md = gr.Markdown(t("presenter_title", LANG_ZH))
                presenter_desc_md = gr.Markdown(t("presenter_desc", LANG_ZH))

                _presenter_default_provider = PROVIDER_NAMES[0] if PROVIDER_NAMES else ""
                _presenter_default_models = PROVIDER_MAP.get(_presenter_default_provider, [])
                _presenter_default_model = _presenter_default_models[0] if _presenter_default_models else ""
                presenter_samples = list_real_chart_samples()
                presenter_default_sample = presenter_samples[0] if presenter_samples else None

                with gr.Row():
                    presenter_analysis_provider = gr.Radio(
                        PROVIDER_NAMES,
                        value=_presenter_default_provider,
                        label=t("presenter_analysis_provider", LANG_ZH),
                        scale=2,
                    )
                    presenter_analysis_model_dd = gr.Dropdown(
                        choices=_presenter_default_models,
                        value=_presenter_default_model,
                        label=t("presenter_analysis_model", LANG_ZH),
                        interactive=True,
                        scale=2,
                    )

                presenter_analysis_model = gr.State(_presenter_default_model)

                def _on_presenter_provider_change(provider):
                    models = PROVIDER_MAP.get(provider, [])
                    return gr.update(choices=models, value=models[0] if models else None)

                def _sync_presenter_model(provider, model):
                    return model or ""

                presenter_analysis_provider.change(
                    _on_presenter_provider_change,
                    inputs=[presenter_analysis_provider],
                    outputs=[presenter_analysis_model_dd],
                )
                presenter_analysis_provider.change(
                    _sync_presenter_model,
                    inputs=[presenter_analysis_provider, presenter_analysis_model_dd],
                    outputs=[presenter_analysis_model],
                )
                presenter_analysis_model_dd.change(
                    _sync_presenter_model,
                    inputs=[presenter_analysis_provider, presenter_analysis_model_dd],
                    outputs=[presenter_analysis_model],
                )

                with gr.Row():
                    presenter_compare_provider = gr.Radio(
                        PROVIDER_NAMES,
                        value=_presenter_default_provider,
                        label=t("presenter_compare_provider", LANG_ZH),
                        scale=2,
                    )
                    presenter_compare_model_dd = gr.Dropdown(
                        choices=_presenter_default_models,
                        value=_presenter_default_model,
                        label=t("presenter_compare_model", LANG_ZH),
                        interactive=True,
                        scale=2,
                    )

                presenter_compare_model = gr.State(_presenter_default_model)

                presenter_compare_provider.change(
                    _on_presenter_provider_change,
                    inputs=[presenter_compare_provider],
                    outputs=[presenter_compare_model_dd],
                )
                presenter_compare_provider.change(
                    _sync_presenter_model,
                    inputs=[presenter_compare_provider, presenter_compare_model_dd],
                    outputs=[presenter_compare_model],
                )
                presenter_compare_model_dd.change(
                    _sync_presenter_model,
                    inputs=[presenter_compare_provider, presenter_compare_model_dd],
                    outputs=[presenter_compare_model],
                )

                with gr.Row():
                    presenter_sample_dd = gr.Dropdown(
                        choices=presenter_samples,
                        value=presenter_default_sample,
                        label=t("presenter_select_sample", LANG_ZH),
                        scale=3,
                    )
                    presenter_run_btn = gr.Button(
                        t("presenter_run_btn", LANG_ZH),
                        variant="primary",
                        scale=1,
                    )

                with gr.Row(equal_height=False):
                    presenter_preview_img = gr.Image(
                        value=get_presenter_sample_image_path(presenter_default_sample) if presenter_default_sample else None,
                        label=t("presenter_preview", LANG_ZH),
                        type="filepath",
                        interactive=False,
                        scale=1,
                    )
                    presenter_sample_info_md = gr.Markdown(
                        value=build_presenter_sample_overview(presenter_default_sample, LANG_ZH) if presenter_default_sample else "",
                        label=t("presenter_sample_info", LANG_ZH),
                    )

                presenter_status_md = gr.Markdown(label=t("presenter_status", LANG_ZH))

                with gr.Row(equal_height=False):
                    presenter_direct_md = gr.Markdown(label=t("presenter_direct_result", LANG_ZH))
                    presenter_geosym_md = gr.Markdown(label=t("presenter_geosym_result", LANG_ZH))

                presenter_pairwise_md = gr.Markdown(label=t("presenter_pairwise_result", LANG_ZH))

                with gr.Accordion(t("presenter_m0_enhanced_title", LANG_ZH), open=True) as presenter_m0_acc:
                    presenter_m0_enhanced_md = gr.Markdown()

                with gr.Accordion(t("presenter_trace_result", LANG_ZH), open=False) as presenter_trace_acc:
                    presenter_trace_md = gr.Markdown()

                presenter_sample_dd.change(
                    get_presenter_sample_image_path,
                    inputs=[presenter_sample_dd],
                    outputs=[presenter_preview_img],
                )
                presenter_sample_dd.change(
                    build_presenter_sample_overview,
                    inputs=[presenter_sample_dd, lang_selector],
                    outputs=[presenter_sample_info_md],
                )

                presenter_run_btn.click(
                    run_presenter_showcase,
                    inputs=[
                        presenter_sample_dd,
                        presenter_analysis_model,
                        presenter_compare_model,
                        lang_selector,
                    ],
                    outputs=[
                        presenter_status_md,
                        presenter_sample_info_md,
                        presenter_direct_md,
                        presenter_geosym_md,
                        presenter_pairwise_md,
                        presenter_trace_md,
                        presenter_m0_enhanced_md,
                    ],
                )

        footer_md = gr.Markdown(t("footer", LANG_ZH))

        # ===== Language change handler =====
        def update_ui_language(lang):
            """Update all UI text based on language selection."""
            updates: list[Any] = []

            # --- Markdown components (value update) ---
            _desc = t('app_desc', lang)
            updates.append(f"# {t('app_title', lang)}\n{_desc}" if _desc else f"# {t('app_title', lang)}")  # title_md
            updates.append(t("rec_tab_title", lang))
            updates.append(t("gen_title", lang))
            updates.append(t("gen_basic", lang))
            updates.append(benchmark_dimensions_md(lang))  # gen_bench_md
            updates.append(t("gen_func_desc", lang))  # gen_func_desc_md
            updates.append(t("gen_page_dim", lang))
            updates.append(t("gen_coord_range", lang))
            updates.append(t("gen_curves_noise", lang))
            updates.append(t("preview_title", lang))
            updates.append(t("facts_tab_title", lang))  # facts_title_md
            updates.append(t("facts_tab_desc", lang))  # facts_desc_md
            updates.append(t("presenter_title", lang))  # presenter_title_md
            updates.append(t("presenter_desc", lang))  # presenter_desc_md
            updates.append(t("conn_test_title", lang))  # conn_title_md
            updates.append(t("conn_overview_title", lang))  # conn_overview_md_title
            updates.append(build_provider_overview(lang))  # conn_overview_md
            updates.append(t("footer", lang))

            # --- Label updates (gr.update) for interactive components ---
            updates.append(gr.update(label=t("gen_count", lang)))
            updates.append(gr.update(label=t("gen_seed", lang)))
            updates.append(gr.update(
                label=t("gen_mode", lang),
                choices=generation_mode_choices(lang),
                value=t("gen_mode_selected", lang),
                interactive=True,
            ))
            updates.append(gr.update(
                label=t("gen_sample_plan", lang),
                choices=sample_plan_choices(lang),
                value=DEFAULT_SAMPLE_PLAN_ID,
                interactive=True,
            ))
            updates.append(gr.update(
                label=t("gen_func_types", lang),
            ))
            updates.append(gr.update(
                label=t("gen_profile", lang),
                choices=profile_choices(lang),
                value=DEFAULT_PROFILE_ID,
                interactive=True,
            ))
            updates.append(gr.update(
                label=t("gen_coord_mode", lang),
                choices=[t("gen_coord_mode_adaptive", lang),
                         t("gen_coord_mode_static", lang),
                         t("gen_coord_mode_manual", lang)],
                value=t("gen_coord_mode_adaptive", lang),
                interactive=True,
            ))
            updates.append(gr.update(label=t("gen_page_width", lang)))
            updates.append(gr.update(label=t("gen_page_height", lang)))
            updates.append(gr.update(label=t("gen_min_intersections", lang)))
            updates.append(gr.update(value=f"**{t('gen_type_cap_title', lang)}**"))
            updates.append(gr.update(value=_build_type_cap_md(lang)))
            updates.append(gr.update(label=t("gen_noise_enabled", lang)))
            updates.append(gr.update(value=t("gen_btn", lang)))  # Button
            updates.append(gr.update(value=t("gen_clear_btn", lang)))  # Clear button
            updates.append(gr.update(label=t("gen_status", lang)))
            updates.append(gr.update(label=t("gen_stats", lang)))
            updates.append(gr.update(label=t("preview_select", lang)))
            updates.append(gr.update(value=t("preview_refresh", lang)))  # Button
            updates.append(gr.update(label=t("preview_img", lang)))
            updates.append(gr.update(label=t("preview_acc_expr", lang)))
            updates.append(gr.update(label=t("preview_acc_props", lang)))
            updates.append(gr.update(label=t("preview_acc_extended", lang)))
            updates.append(gr.update(label=t("preview_acc_ints", lang)))
            updates.append(gr.update(label=t("preview_acc_tasks", lang)))
            updates.append(gr.update(label=t("preview_acc_meta", lang)))
            updates.append(gr.update(label=t("facts_select_image", lang)))  # facts_image_dd
            updates.append(gr.update(value=t("facts_run_btn", lang)))  # facts_run_btn
            updates.append(gr.update(label=t("facts_preview", lang)))  # facts_preview_img
            updates.append(gr.update(label=t("presenter_analysis_provider", lang)))  # presenter_analysis_provider
            updates.append(gr.update(label=t("presenter_analysis_model", lang)))  # presenter_analysis_model_dd
            updates.append(gr.update(label=t("presenter_compare_provider", lang)))  # presenter_compare_provider
            updates.append(gr.update(label=t("presenter_compare_model", lang)))  # presenter_compare_model_dd
            updates.append(gr.update(label=t("presenter_select_sample", lang)))  # presenter_sample_dd
            updates.append(gr.update(value=t("presenter_run_btn", lang)))  # presenter_run_btn
            updates.append(gr.update(label=t("presenter_preview", lang)))  # presenter_preview_img
            updates.append(gr.update(label=t("presenter_sample_info", lang)))  # presenter_sample_info_md
            updates.append(gr.update(label=t("presenter_status", lang)))  # presenter_status_md
            updates.append(gr.update(label=t("presenter_direct_result", lang)))  # presenter_direct_md
            updates.append(gr.update(label=t("presenter_geosym_result", lang)))  # presenter_geosym_md
            updates.append(gr.update(label=t("presenter_pairwise_result", lang)))  # presenter_pairwise_md
            # --- Coordinate range sliders ---
            updates.append(gr.update(label=t("gen_x_min", lang)))
            updates.append(gr.update(label=t("gen_x_max", lang)))
            updates.append(gr.update(label=t("gen_y_min", lang)))
            updates.append(gr.update(label=t("gen_y_max", lang)))

            # --- Math Reasoning tab ---
            updates.append(gr.update(value=t("reason_title", lang)))  # reason_title_md
            updates.append(gr.update(value=t("reason_conditions_desc", lang)))  # reason_conditions_md
            updates.append(gr.update(label=t("reason_provider", lang)))  # reason_provider
            updates.append(gr.update(label=t("reason_model", lang)))  # reason_model_dd
            updates.append(gr.update(label=t("judge_provider", lang)))  # judge_provider
            updates.append(gr.update(label=t("judge_model", lang)))  # judge_model_dd
            updates.append(gr.update(label=t("reason_select_image", lang)))  # reason_single_image
            updates.append(gr.update(label=t("reason_preview", lang)))  # reason_preview_img
            updates.append(gr.update(value=t("reason_btn_run_all", lang)))  # reason_all_btn
            updates.append(gr.update(value=t("reason_btn_retry", lang)))  # reason_retry_btn
            updates.append(gr.update(value=t("reason_btn_rerun_all", lang)))  # reason_rerun_btn
            updates.append(gr.update(value=t("reason_btn_clear_cache", lang)))  # reason_clear_btn
            updates.append(gr.update(value=t("reason_btn_clear_all_cache", lang)))  # reason_clear_all_btn
            updates.append(gr.update(label=t("reason_select_rerun_condition", lang)))  # reason_rerun_condition_dd
            updates.append(gr.update(value=t("reason_btn_rerun_selected", lang)))  # reason_rerun_condition_btn
            # --- Connection Test tab ---
            updates.append(gr.update(label=t("conn_provider", lang)))  # conn_provider
            updates.append(gr.update(label=t("conn_model", lang)))  # conn_model_dd
            updates.append(gr.update(value=t("conn_test_btn", lang)))  # conn_test_btn
            updates.append(gr.update(label=t("conn_result_label", lang)))  # conn_result_md

            # --- Tab / Accordion label updates ---
            updates.append(gr.update(label=t("tab_recognizer", lang)))
            updates.append(gr.update(label=t("tab_generation", lang)))
            updates.append(gr.update(label=t("tab_reasoning", lang)))
            updates.append(gr.update(label=t("tab_facts_accuracy", lang)))
            updates.append(gr.update(label=t("tab_presenter", lang)))
            updates.append(gr.update(label=t("tab_conn_test", lang)))
            updates.append(gr.update(label=t("gen_advanced", lang)))  # acc_gen_advanced
            updates.append(gr.update(label=t("presenter_trace_result", lang)))  # presenter_trace_acc
            updates.append(gr.update(label=t("presenter_m0_enhanced_title", lang)))  # presenter_m0_acc

            # Update global language state (used by recognizer handlers)
            set_lang(lang)

            # Recognizer sub-UI components
            rec_updates = update_recognizer_language(
                _rec_md_components, _rec_interactive_components, lang
            )
            updates.extend(rec_updates)

            return updates

        _lang_outputs: list[Any] = [
            # --- Markdown components ---
            title_md,
            rec_tab_title_md,
            gen_title_md,
            gen_basic_md,
            gen_bench_md,
            gen_func_desc_md,
            gen_page_md,
            gen_coord_md,
            gen_curves_md,
            preview_title_md,
            facts_title_md, facts_desc_md,
            presenter_title_md, presenter_desc_md,
            conn_title_md,
            conn_overview_md_title,
            conn_overview_md,
            footer_md,
            # --- Interactive component label updates ---
            gen_count, gen_seed, gen_mode,
            gen_sample_plan,
            gen_func_types, gen_profile, gen_coord_mode,
            page_width, page_height, min_intersections,
            type_cap_title, type_cap_md,
            noise_enabled,
            gen_btn, gen_clear_btn,
            gen_status, gen_stats,
            dataset_dropdown, refresh_btn,
            preview_img,
            pv_acc_expr, pv_acc_props, pv_acc_extended,
            pv_acc_ints, pv_acc_tasks, pv_acc_meta,
            facts_image_dd, facts_run_btn, facts_preview_img,
            presenter_analysis_provider, presenter_analysis_model_dd,
            presenter_compare_provider, presenter_compare_model_dd,
            presenter_sample_dd, presenter_run_btn,
            presenter_preview_img, presenter_sample_info_md,
            presenter_status_md, presenter_direct_md, presenter_geosym_md,
            presenter_pairwise_md,
            # --- Coordinate range sliders ---
            x_min, x_max, y_min, y_max,
            # --- Math Reasoning tab ---
            reason_title_md, reason_conditions_md,
            reason_provider, reason_model_dd,
            judge_provider, judge_model_dd,
            reason_single_image, reason_preview_img, reason_all_btn, reason_retry_btn,
            reason_rerun_btn, reason_clear_btn, reason_clear_all_btn,
            reason_rerun_condition_dd, reason_rerun_condition_btn,
            # --- Connection Test tab ---
            conn_provider, conn_model_dd, conn_test_btn, conn_result_md,
            # --- Tabs and Accordions ---
            tab_recognizer, tab_gen,
            tab_reasoning,
            tab_facts_accuracy,
            tab_presenter,
            tab_conn_test,
            acc_gen_advanced,
            presenter_trace_acc,
            presenter_m0_acc,
        ] + [comp for _, comp in _rec_md_components] + [comp for _, comp, _ in _rec_interactive_components]

        # Build JS for Tab / Accordion label refresh (Gradio doesn't
        # visually re-render these from gr.update alone).
        _tab_acc_keys = {
            "tabs": [
                "tab_recognizer", "tab_generation",
                "tab_reasoning",
                "tab_facts_accuracy",
                "tab_presenter",
                "tab_conn_test",
                "reason_cond_e0", "reason_cond_e1", "reason_cond_e2",
                "reason_cond_e3", "reason_cond_e4", "reason_cond_e5",
                "s0_tab_result", "s0_tab_debug", "s0_tab_ocr_info",
                "s0_tab_ocr_raw", "s0_tab_ocr_vis", "s0_tab_gemini",
                "s1_tab_final", "s1_tab_edges", "s1_tab_lines",
                "s1_tab_dashed", "s1_tab_merged",
                "s2_tab_intermediate", "s2_tab_curve", "s2_tab_grid_coords",
                "s2_tab_annotated", "s2_tab_raw_table", "s2_tab_filtered_table", "s2_tab_ruler",
                "s3_tab_ai", "s3_tab_raw", "s3_tab_table",
            ],
            "accs": [
                "gen_advanced",
                "presenter_trace_result",
                "presenter_m0_enhanced_title",
                "preview_acc_expr", "preview_acc_props", "preview_acc_extended",
                "preview_acc_ints", "preview_acc_tasks", "preview_acc_meta",
                "s0_advanced",
                "s1_curve_params", "s1_dashed_params", "s1_rot_params",
                "s1_grid_params", "s1_merge_params",
                "s2_curve_extract", "s2_filter_params",
            ],
        }
        import json as _json
        _tab_pairs = _json.dumps(
            [[t(k, LANG_ZH), t(k, LANG_EN)] for k in _tab_acc_keys["tabs"]],
            ensure_ascii=False,
        )
        _acc_pairs = _json.dumps(
            [[t(k, LANG_ZH), t(k, LANG_EN)] for k in _tab_acc_keys["accs"]],
            ensure_ascii=False,
        )

        _TAB_REFRESH_JS = f"""(lang) => {{
            const isEn = (lang === 'English');
            const tabs = {_tab_pairs};
            const accs = {_acc_pairs};
            document.querySelectorAll('button[role="tab"]').forEach(btn => {{
                const txt = btn.textContent.trim();
                for (const [zh, en] of tabs) {{
                    if (txt === zh || txt === en) {{
                        btn.textContent = isEn ? en : zh;
                        break;
                    }}
                }}
            }});
            document.querySelectorAll('.accordion > .label-wrap > span:first-child, .accordion .label-wrap span.svelte-1w6vloh').forEach(span => {{
                const txt = span.textContent.trim();
                for (const [zh, en] of accs) {{
                    if (txt === zh || txt === en) {{
                        span.textContent = isEn ? en : zh;
                        break;
                    }}
                }}
            }});
            if (!document.querySelector('.accordion > .label-wrap > span:first-child')) {{
                document.querySelectorAll('button.label-wrap, div.label-wrap').forEach(wrap => {{
                    const span = wrap.querySelector('span');
                    if (!span) return;
                    const txt = span.textContent.trim();
                    for (const [zh, en] of accs) {{
                        if (txt === zh || txt === en) {{
                            span.textContent = isEn ? en : zh;
                            break;
                        }}
                    }}
                }});
            }}
        }}"""

        _all_image_dropdowns = [dataset_dropdown, reason_single_image, facts_image_dd]
        refresh_btn.click(refresh_dataset, outputs=_all_image_dropdowns)
        _gen_event.then(refresh_dataset, outputs=_all_image_dropdowns)

        lang_selector.change(
            update_ui_language,
            inputs=[lang_selector],
            outputs=_lang_outputs,
        ).then(
            fn=None,
            inputs=[lang_selector],
            js=_TAB_REFRESH_JS,
        )

    return app

