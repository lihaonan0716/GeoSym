"""
Centralized internationalization (i18n) module for the Experiment project.

All user-facing text is defined here in both English and Chinese.

Usage:
    from i18n import t, LANG_EN, LANG_ZH
    label = t("gen_btn", lang)  # returns the localized string
"""

from typing import Dict

LANG_EN = "English"
LANG_ZH = "中文"

TEXTS: Dict[str, Dict[str, str]] = {
    # =====================================================================
    # App-level
    # =====================================================================
    "app_title": {
        LANG_EN: "🔬 GeoSym",
        LANG_ZH: "🔬 GeoSym",
    },
    "app_desc": {
        LANG_EN: "",
        LANG_ZH: "",
    },
    "footer": {
        LANG_EN: "",
        LANG_ZH: "",
    },

    # =====================================================================
    # Tab names
    # =====================================================================
    "tab_recognizer": {
        LANG_EN: "🔬 GeoSym Function-Specialized M0 Demo",
        LANG_ZH: "🔬 GeoSym函数特化M0演示",
    },
    "tab_generation": {
        LANG_EN: "📷 Benchmark Generation",
        LANG_ZH: "📷 Benchmark 生成",
    },
    "tab_reasoning": {
        LANG_EN: "🧠 GeoSym Full Function Reasoning",
        LANG_ZH: "🧠 GeoSym完整函数推理",
    },
    "tab_facts_accuracy": {
        LANG_EN: "📐 GeoSym Function-Specialized M0 Test",
        LANG_ZH: "📐 GeoSym函数特化M0测试",
    },
    "tab_presenter": {
        LANG_EN: "🎬 Real Chart Presenter",
        LANG_ZH: "🎬 真实图像 Presenter",
    },
    "tab_facts_accuracy": {
        LANG_EN: "📐 GeoSym Function-Specialized M0 Test",
        LANG_ZH: "📐 GeoSym函数特化M0测试",
    },
    "reason_title": {
        LANG_EN: "**Structured Visual-Symbolic Fusion: Six-Condition Reasoning Comparison**",
        LANG_ZH: "**结构化视觉-符号融合：六条件推理对比**",
    },
    "reason_conditions_title": {
        LANG_EN: "Experimental Conditions (E0-E5)",
        LANG_ZH: "实验条件说明（E0-E5）",
    },
    "reason_conditions_desc": {
        LANG_EN: (
            "This page follows the **six-condition unified experiment matrix**:\n\n"
            "| Condition | Description |\n"
            "|---|---|\n"
            "| **E0** | Direct visual QA |\n"
            "| **E1** | Direct visual QA with chain-of-thought prompting |\n"
            "| **E2** | Direct visual QA with self-consistency aggregation |\n"
            "| **E3** | OCR + rule evidence assisted QA |\n"
            "| **E4** | OCR + rule evidence with chain-of-thought prompting |\n"
            "| **E5** | Full GeoSym pipeline (M0+M1+M2+M3) |\n\n"
            "**Run all:** E0-E5"
        ),
        LANG_ZH: (
            "本页面采用 **六条件统一实验矩阵**：\n\n"
            "| 条件 | 说明 |\n"
            "|---|---|\n"
            "| **E0** | 直接视觉问答 |\n"
            "| **E1** | 直接视觉问答 + CoT |\n"
            "| **E2** | 直接视觉问答 + 自一致性聚合 |\n"
            "| **E3** | OCR + 规则证据辅助问答 |\n"
            "| **E4** | OCR + 规则证据 + CoT |\n"
            "| **E5** | 完整 GeoSym 管线（M0+M1+M2+M3） |\n\n"
            "**运行全部：** E0-E5"
        ),
    },
    "reason_select_model": {LANG_EN: "Model", LANG_ZH: "模型"},
    "reason_provider": {LANG_EN: "Provider", LANG_ZH: "模型供应商"},
    "reason_model": {LANG_EN: "Model", LANG_ZH: "模型"},
    "judge_provider": {LANG_EN: "Judge Provider", LANG_ZH: "评判模型供应商"},
    "judge_model": {LANG_EN: "Judge Model", LANG_ZH: "评判模型"},
    "reason_select_image": {LANG_EN: "Select Image", LANG_ZH: "选择图像"},
    "reason_btn_run_all": {LANG_EN: "Run All 6 Conditions", LANG_ZH: "运行全部 6 个条件"},
    "reason_btn_retry": {LANG_EN: "Re-run Failed Conditions", LANG_ZH: "重跑失败条件"},
    "reason_preview": {LANG_EN: "Image Preview", LANG_ZH: "图像预览"},
    "reason_ground_truth": {LANG_EN: "Ground Truth", LANG_ZH: "标准答案"},
    "reason_path_a": {LANG_EN: "E0: DirectQA", LANG_ZH: "E0：直接视觉问答"},
    "reason_path_b": {LANG_EN: "E5: FullGeoSym", LANG_ZH: "E5：完整 GeoSym"},
    "reason_summary_table": {LANG_EN: "Score Comparison", LANG_ZH: "评分对比"},
    "reason_cond_e0": {LANG_EN: "E0: DirectQA", LANG_ZH: "E0：直接视觉问答"},
    "reason_cond_e1": {LANG_EN: "E1: DirectQA-CoT", LANG_ZH: "E1：直接视觉问答 + CoT"},
    "reason_cond_e2": {LANG_EN: "E2: DirectQA-SelfConsistency", LANG_ZH: "E2：直接视觉问答 + 自一致性"},
    "reason_cond_e3": {LANG_EN: "E3: OCRRulesQA", LANG_ZH: "E3：OCR + 规则问答"},
    "reason_cond_e4": {LANG_EN: "E4: OCRRules-CoT", LANG_ZH: "E4：OCR + 规则问答 + CoT"},
    "reason_cond_e5": {LANG_EN: "E5: FullGeoSym", LANG_ZH: "E5：完整 GeoSym"},
    "reason_cond_e5_raw": {LANG_EN: "E5_raw: GeoSym-Raw", LANG_ZH: "E5_raw：GeoSym 原始坐标"},
    "reason_ablation_title": {
        LANG_EN: "Retired Ablation Labels",
        LANG_ZH: "已弃用的消融标签",
    },
    "reason_ablation_desc": {
        LANG_EN: "The old factorized naming has been retired. Use the unified E0-E5 matrix instead.",
        LANG_ZH: "旧的分解式命名已废弃，请统一使用 E0-E5 实验矩阵。",
    },
    "reason_abl_btn_run": {LANG_EN: "Deprecated", LANG_ZH: "已弃用"},
    "reason_abl_btn_retry": {LANG_EN: "Deprecated", LANG_ZH: "已弃用"},
    "reason_abl_btn_rerun": {LANG_EN: "🔄 Re-run All (Ignore Cache)", LANG_ZH: "🔄 全部重新实验"},
    "reason_btn_rerun_all": {LANG_EN: "🔄 Re-run All (Ignore Cache)", LANG_ZH: "🔄 全部重新实验"},
    "reason_btn_clear_cache": {LANG_EN: "🗑 Clear Current Image Cache", LANG_ZH: "🗑 清空当前图像缓存"},
    "reason_btn_clear_all_cache": {LANG_EN: "🗑 Clear All Reasoning Cache", LANG_ZH: "🗑 清空全部推理缓存"},
    "reason_select_rerun_condition": {LANG_EN: "Condition to Re-run", LANG_ZH: "选择要重跑的条件"},
    "reason_btn_rerun_selected": {LANG_EN: "🔄 Re-run Selected Condition", LANG_ZH: "🔄 重跑所选条件"},
    "reason_abl_c1": {LANG_EN: "Deprecated", LANG_ZH: "已弃用"},
    "reason_abl_c1a": {LANG_EN: "Deprecated", LANG_ZH: "已弃用"},
    "reason_abl_c1b": {LANG_EN: "Deprecated", LANG_ZH: "已弃用"},
    "reason_abl_c2": {LANG_EN: "Deprecated", LANG_ZH: "已弃用"},
    "reason_judge_label": {LANG_EN: "Judge LLM", LANG_ZH: "Judge 模型"},
    "reason_weights_title": {LANG_EN: "Scoring Weights", LANG_ZH: "评分权重配置"},
    "reason_w_expression": {LANG_EN: "Expression", LANG_ZH: "表达式"},
    "reason_w_domain": {LANG_EN: "Domain", LANG_ZH: "定义域"},
    "reason_w_range": {LANG_EN: "Range", LANG_ZH: "值域"},
    "reason_w_predictions": {LANG_EN: "Value Prediction", LANG_ZH: "函数值预测"},
    "reason_w_rate": {LANG_EN: "Rate of Change", LANG_ZH: "变化率"},

    # =====================================================================
    # Presenter tab
    # =====================================================================
    "presenter_title": {
        LANG_EN: "## Real-Chart Showcase: Direct LLM vs GeoSym",
        LANG_ZH: "## 真实图像展示：纯 LLM vs GeoSym",
    },
    "presenter_desc": {
        LANG_EN: "",
        LANG_ZH: "",
    },
    "presenter_analysis_provider": {
        LANG_EN: "Analysis Provider",
        LANG_ZH: "主分析模型供应商",
    },
    "presenter_analysis_model": {
        LANG_EN: "Analysis Model",
        LANG_ZH: "主分析模型",
    },
    "presenter_compare_provider": {
        LANG_EN: "Comparison Provider",
        LANG_ZH: "比较模型供应商",
    },
    "presenter_compare_model": {
        LANG_EN: "Comparison Model",
        LANG_ZH: "比较模型",
    },
    "presenter_select_sample": {
        LANG_EN: "Select Real Chart Sample",
        LANG_ZH: "选择真实图像样例",
    },
    "presenter_run_btn": {
        LANG_EN: "Run Live Showcase",
        LANG_ZH: "实时运行展示",
    },
    "presenter_preview": {
        LANG_EN: "Sample Preview",
        LANG_ZH: "样例预览",
    },
    "presenter_sample_info": {
        LANG_EN: "Sample Information",
        LANG_ZH: "样例信息",
    },
    "presenter_reference": {
        LANG_EN: "CSV Reference",
        LANG_ZH: "CSV 参考",
    },
    "presenter_check": {
        LANG_EN: "Automatic Check",
        LANG_ZH: "自动核对",
    },
    "presenter_status": {
        LANG_EN: "Run Status",
        LANG_ZH: "运行状态",
    },
    "presenter_direct_result": {
        LANG_EN: "Direct LLM Result",
        LANG_ZH: "纯 LLM 结果",
    },
    "presenter_geosym_result": {
        LANG_EN: "GeoSym Result",
        LANG_ZH: "GeoSym结果",
    },
    "presenter_pairwise_result": {
        LANG_EN: "Pairwise Comparison",
        LANG_ZH: "成对比较结论",
    },
    "presenter_trace_result": {
        LANG_EN: "GeoSym Internal Trace",
        LANG_ZH: "GeoSym 内部过程",
    },
    "presenter_m0_enhanced_title": {
        LANG_EN: "M0 Enhanced: OCR Ruler + LLM Locate + Numerical Measure",
        LANG_ZH: "M0 数值增强: OCR标尺 + LLM定位 + 数值计算",
    },
    "presenter_ruler_detail": {
        LANG_EN: "Phase 1: Coordinate Ruler (OCR)",
        LANG_ZH: "Phase 1: 坐标标尺 (OCR)",
    },
    "presenter_locate_detail": {
        LANG_EN: "Phase 2: LLM Locate (Pixel Coordinates)",
        LANG_ZH: "Phase 2: LLM 定位 (像素坐标)",
    },
    "presenter_measure_detail": {
        LANG_EN: "Phase 3: Numerical Measurement (pixel → value)",
        LANG_ZH: "Phase 3: 数值计算 (像素 → 数值)",
    },
    "presenter_computation_trace": {
        LANG_EN: "Computation Trace",
        LANG_ZH: "计算链路追踪",
    },

    # =====================================================================
    # Facts accuracy tab
    # =====================================================================
    "facts_tab_title": {
        LANG_EN: "## GeoSym Function-Specialized M0 Test",
        LANG_ZH: "## GeoSym函数特化M0测试",
    },
    "facts_tab_desc": {
        LANG_EN: "Evaluate GeoSym function-specialized M0 extraction quality (coordinates/intercepts/extrema/visible-range monotonicity).",
        LANG_ZH: "评估 GeoSym函数特化M0 提取质量（坐标/截距/极值/可见范围单调性）。",
    },
    "facts_select_image": {
        LANG_EN: "Select Image",
        LANG_ZH: "选择图片",
    },
    "facts_run_btn": {
        LANG_EN: "Run Diagnosis",
        LANG_ZH: "运行诊断",
    },
    "facts_preview": {
        LANG_EN: "Image Preview",
        LANG_ZH: "图片预览",
    },
    "facts_summary_title": {
        LANG_EN: "### GeoSym Function-Specialized M0 Accuracy",
        LANG_ZH: "### GeoSym函数特化M0 准确性",
    },
    "facts_err_select_image": {
        LANG_EN: "Please select an image before running diagnosis.",
        LANG_ZH: "请选择图片后再运行。",
    },
    "facts_err_image_not_found": {
        LANG_EN: "Image not found: `{image}`",
        LANG_ZH: "未找到图片：`{image}`",
    },
    "facts_err_gt_not_found": {
        LANG_EN: "Ground truth not found: `{image}`",
        LANG_ZH: "未找到真值：`{image}`",
    },
    "facts_summary_image": {
        LANG_EN: "Image",
        LANG_ZH: "图片",
    },
    "facts_summary_avg": {
        LANG_EN: "Overall (4-item average)",
        LANG_ZH: "综合分 (4项均值)",
    },
    "facts_item_coordinates": {
        LANG_EN: "Coordinates",
        LANG_ZH: "坐标",
    },
    "facts_item_intercepts": {
        LANG_EN: "Intercepts",
        LANG_ZH: "截距",
    },
    "facts_item_extrema": {
        LANG_EN: "Extrema",
        LANG_ZH: "极值",
    },
    "facts_item_monotonicity": {
        LANG_EN: "Visible-range Monotonicity",
        LANG_ZH: "可见范围单调性",
    },
    "facts_detail_title": {
        LANG_EN: "### Prediction vs Ground Truth",
        LANG_ZH: "### 预测 vs 真值",
    },
    "facts_raw_title": {
        LANG_EN: "### Raw Diagnosis Data",
        LANG_ZH: "### 原始诊断数据",
    },
    "facts_label_pred": {
        LANG_EN: "Pred",
        LANG_ZH: "预测",
    },
    "facts_label_gt": {
        LANG_EN: "GT",
        LANG_ZH: "真值",
    },
    "facts_label_details": {
        LANG_EN: "Details",
        LANG_ZH: "说明",
    },
    "facts_diag_gt_total": {
        LANG_EN: "GT total",
        LANG_ZH: "真值总数",
    },
    "facts_diag_pred_total": {
        LANG_EN: "Pred total",
        LANG_ZH: "预测总数",
    },
    "facts_diag_matched": {
        LANG_EN: "Matched",
        LANG_ZH: "命中",
    },
    "facts_diag_missed": {
        LANG_EN: "Missed GT",
        LANG_ZH: "漏检",
    },
    "facts_diag_extra": {
        LANG_EN: "Extra Pred",
        LANG_ZH: "误检",
    },
    "facts_diag_type_ok": {
        LANG_EN: "Type-correct matches",
        LANG_ZH: "类型正确命中",
    },
    "facts_diag_matched_pairs": {
        LANG_EN: "Matched pairs",
        LANG_ZH: "命中配对",
    },
    "facts_diag_missed_points": {
        LANG_EN: "Missed GT points",
        LANG_ZH: "漏检真值点",
    },
    "facts_diag_extra_points": {
        LANG_EN: "Extra predicted points",
        LANG_ZH: "误检预测点",
    },
    "facts_diag_none": {
        LANG_EN: "none",
        LANG_ZH: "无",
    },

    # =====================================================================
    # Facts accuracy tab
    # =====================================================================
    "facts_tab_title": {
        LANG_EN: "## GeoSym Function-Specialized M0 Test",
        LANG_ZH: "## GeoSym函数特化M0测试",
    },
    "facts_tab_desc": {
        LANG_EN: "Evaluate GeoSym function-specialized M0 extraction quality (coordinates/intercepts/extrema/visible-range monotonicity).",
        LANG_ZH: "评估 GeoSym函数特化M0 提取质量（坐标/截距/极值/可见范围单调性）。",
    },
    "facts_select_image": {
        LANG_EN: "Select Image",
        LANG_ZH: "选择图片",
    },
    "facts_run_btn": {
        LANG_EN: "Run Diagnosis",
        LANG_ZH: "运行诊断",
    },
    "facts_preview": {
        LANG_EN: "Image Preview",
        LANG_ZH: "图片预览",
    },
    "facts_summary_title": {
        LANG_EN: "### GeoSym Function-Specialized M0 Accuracy",
        LANG_ZH: "### GeoSym函数特化M0 准确性",
    },
    "facts_err_select_image": {
        LANG_EN: "Please select an image before running diagnosis.",
        LANG_ZH: "请选择图片后再运行。",
    },
    "facts_err_image_not_found": {
        LANG_EN: "Image not found: `{image}`",
        LANG_ZH: "未找到图片：`{image}`",
    },
    "facts_err_gt_not_found": {
        LANG_EN: "Ground truth not found: `{image}`",
        LANG_ZH: "未找到真值：`{image}`",
    },
    "facts_summary_image": {
        LANG_EN: "Image",
        LANG_ZH: "图片",
    },
    "facts_summary_avg": {
        LANG_EN: "Overall (4-item average)",
        LANG_ZH: "综合分 (4项均值)",
    },
    "facts_item_coordinates": {
        LANG_EN: "Coordinates",
        LANG_ZH: "坐标",
    },
    "facts_item_intercepts": {
        LANG_EN: "Intercepts",
        LANG_ZH: "截距",
    },
    "facts_item_extrema": {
        LANG_EN: "Extrema",
        LANG_ZH: "极值",
    },
    "facts_item_monotonicity": {
        LANG_EN: "Visible-range Monotonicity",
        LANG_ZH: "可见范围单调性",
    },
    "facts_detail_title": {
        LANG_EN: "### Prediction vs Ground Truth",
        LANG_ZH: "### 预测 vs 真值",
    },
    "facts_raw_title": {
        LANG_EN: "### Raw Diagnosis Data",
        LANG_ZH: "### 原始诊断数据",
    },

    # =====================================================================
    # Recognizer tab
    # =====================================================================
    "rec_tab_title": {
        LANG_EN: "### GeoSym Function-Specialized M0 Demo\nUpload a problem image and run the full M0 extraction pipeline interactively.",
        LANG_ZH: "### GeoSym函数特化M0演示\n上传题目图片，交互式运行完整的 M0 提取流程。",
    },

    # =====================================================================
    # Generation tab
    # =====================================================================
    "gen_title": {
        LANG_EN: "### Generate Test Images\nCreate new images with customizable parameters for evaluation.",
        LANG_ZH: "### 生成测试图像\n使用可自定义的参数创建新图像用于评估。",
    },
    "gen_basic": {
        LANG_EN: "**Basic Settings**",
        LANG_ZH: "**基本设置**",
    },
    "gen_count": {
        LANG_EN: "Uniform per-subset count",
        LANG_ZH: "统一每子集数量",
    },
    "gen_sample_plan": {
        LANG_EN: "Sample allocation strategy",
        LANG_ZH: "样本配额策略",
    },
    "gen_sample_plan_uniform": {
        LANG_EN: "Uniform count for every function type",
        LANG_ZH: "所有函数类型使用统一数量",
    },
    "gen_sample_plan_recommended": {
        LANG_EN: "Recommended benchmark allocation",
        LANG_ZH: "推荐 Benchmark 配额",
    },
    "gen_sample_plan_preview_title": {
        LANG_EN: "Per-function sample plan",
        LANG_ZH: "各函数类型样本配额",
    },
    "gen_sample_plan_active": {
        LANG_EN: "Active allocation",
        LANG_ZH: "当前配额策略",
    },
    "gen_sample_plan_uniform_note": {
        LANG_EN: "Uniform mode: every included subset uses `{count}` images.",
        LANG_ZH: "统一模式：每个参与生成的子集都使用 `{count}` 张图像。",
    },
    "gen_sample_plan_recommended_note": {
        LANG_EN: "Recommended mode uses the tested per-function allocation and ignores the uniform-count slider.",
        LANG_ZH: "推荐模式使用经过测试的按函数类型配额，并忽略上方的统一数量滑块。",
    },
    "gen_sample_plan_total": {
        LANG_EN: "Estimated images for current selection",
        LANG_ZH: "当前选择预计生成图像数",
    },
    "gen_seed": {
        LANG_EN: "Random seed (0 for random)",
        LANG_ZH: "随机种子（0为随机）",
    },
    "gen_mode": {
        LANG_EN: "Generation mode",
        LANG_ZH: "生成模式",
    },
    "gen_mode_selected": {
        LANG_EN: "Selected types (append)",
        LANG_ZH: "已选类型（追加）",
    },
    "gen_mode_all_subsets": {
        LANG_EN: "All 36 subsets (batch)",
        LANG_ZH: "全部 36 子集（批量）",
    },
    "gen_clear_btn": {
        LANG_EN: "🗑️ Clear All Data",
        LANG_ZH: "🗑️ 清空全部数据",
    },
    "gen_clear_confirm": {
        LANG_EN: "✅ All images and ground truth cleared.",
        LANG_ZH: "✅ 已清空全部图像和真值数据。",
    },
    "gen_page_dim": {
        LANG_EN: "**Page Dimensions**",
        LANG_ZH: "**页面尺寸**",
    },
    "gen_page_width": {
        LANG_EN: "Page width (px)",
        LANG_ZH: "页面宽度（像素）",
    },
    "gen_page_height": {
        LANG_EN: "Page height (px)",
        LANG_ZH: "页面高度（像素）",
    },
    "gen_coord_range": {
        LANG_EN: "**Coordinate Range**",
        LANG_ZH: "**坐标范围**",
    },
    "gen_curves_noise": {
        LANG_EN: "**Degradation Settings**",
        LANG_ZH: "**退化设置**",
    },
    "gen_noise_enabled": {
        LANG_EN: "Enable profile degradation",
        LANG_ZH: "启用档位退化",
    },
    "gen_min_intersections": {
        LANG_EN: "Min grid intersections per image",
        LANG_ZH: "每张图最少网格交点数",
    },
    "gen_type_cap_title": {
        LANG_EN: "Per-type intersection caps (auto-downgrade)",
        LANG_ZH: "各函数类型交点上限（自动降级）",
    },
    "gen_x_min": {LANG_EN: "X min", LANG_ZH: "X 最小值"},
    "gen_x_max": {LANG_EN: "X max", LANG_ZH: "X 最大值"},
    "gen_y_min": {LANG_EN: "Y min", LANG_ZH: "Y 最小值"},
    "gen_y_max": {LANG_EN: "Y max", LANG_ZH: "Y 最大值"},

    # --- Benchmark dimensions ---
    "gen_bench_dims": {
        LANG_EN: "**Benchmark Dimensions** (8 types × 4 profiles = 32 subsets)",
        LANG_ZH: "**Benchmark 维度**（8 种函数 × 4 种档位 = 32 子集）",
    },
    "gen_func_types": {LANG_EN: "Function Types", LANG_ZH: "函数类型"},
    "gen_func_desc": {
        LANG_EN: (
            "| Type | Expression | Category |\n"
            "|------|-----------|----------|\n"
            "| linear | *y = ax + b* | elementary |\n"
            "| quadratic | *y = a(x−h)² + k* | elementary |\n"
            "| absolute | *y = a\\|x−h\\| + k* | elementary |\n"
            "| cubic | *y = a(x−h)³ + b(x−h) + k* | polynomial |\n"
            "| piecewise_linear | *y = s₁(x−h)+k / s₂(x−h)+k* | polynomial |\n"
            "| exponential | *y = a·b^(x−h) + k* | transcendental |\n"
            "| logarithmic | *y = a·log₂(x−h) + k* | transcendental |\n"
            "| rational | *y = a/(x−h) + k* | special |"
        ),
        LANG_ZH: (
            "| 类型 | 表达式 | 分类 |\n"
            "|------|--------|------|\n"
            "| linear | *y = ax + b* | 初等 |\n"
            "| quadratic | *y = a(x−h)² + k* | 初等 |\n"
            "| absolute | *y = a\\|x−h\\| + k* | 初等 |\n"
            "| cubic | *y = a(x−h)³ + b(x−h) + k* | 多项式 |\n"
            "| piecewise_linear | *y = s₁(x−h)+k / s₂(x−h)+k* | 多项式 |\n"
            "| exponential | *y = a·b^(x−h) + k* | 超越函数 |\n"
            "| logarithmic | *y = a·log₂(x−h) + k* | 超越函数 |\n"
            "| rational | *y = a/(x−h) + k* | 特殊 |"
        ),
    },
    "gen_profile": {LANG_EN: "Benchmark Profile", LANG_ZH: "基准测试档位"},
    "gen_profile_clean_render": {
        LANG_EN: "Clean Render — no synthetic degradation",
        LANG_ZH: "Clean Render — 无合成退化",
    },
    "gen_coord_mode": {
        LANG_EN: "Coordinate Range Mode",
        LANG_ZH: "坐标范围模式",
    },
    "gen_coord_mode_manual": {
        LANG_EN: "Manual (use slider values)",
        LANG_ZH: "手动（使用滑块值）",
    },
    "gen_coord_mode_static": {
        LANG_EN: "Static Recommended (per function type)",
        LANG_ZH: "静态推荐（按函数类型）",
    },
    "gen_coord_mode_adaptive": {
        LANG_EN: "Adaptive Auto-fit (parameter-aware)",
        LANG_ZH: "自适应微调（参数感知）",
    },
    "gen_advanced": {
        LANG_EN: "Advanced Settings (Override)",
        LANG_ZH: "高级设置（手动覆盖）",
    },
    "preview_expressions": {
        LANG_EN: "Function Expressions",
        LANG_ZH: "函数表达式",
    },

    "gen_btn": {
        LANG_EN: "🚀 Generate Images",
        LANG_ZH: "🚀 生成图像",
    },
    "gen_status": {
        LANG_EN: "Status",
        LANG_ZH: "状态",
    },
    "gen_stats": {
        LANG_EN: "Statistics",
        LANG_ZH: "统计信息",
    },
    "gen_preview": {
        LANG_EN: "Last generated image",
        LANG_ZH: "最后生成的图像",
    },

    # =====================================================================
    # Dataset preview
    # =====================================================================
    "preview_title": {
        LANG_EN: "### Browse Dataset\nPreview generated images and all ground truth annotations.",
        LANG_ZH: "### 浏览数据集\n预览生成的图像及其完整真值标注。",
    },
    "preview_select": {
        LANG_EN: "Select image",
        LANG_ZH: "选择图像",
    },
    "preview_refresh": {
        LANG_EN: "🔄 Refresh",
        LANG_ZH: "🔄 刷新",
    },
    "preview_img": {
        LANG_EN: "Original Image",
        LANG_ZH: "原始图像",
    },
    "preview_acc_expr": {
        LANG_EN: "Function Expressions",
        LANG_ZH: "函数表达式",
    },
    "preview_acc_props": {
        LANG_EN: "Math Properties (Domain / Range / Zeros)",
        LANG_ZH: "数学属性（定义域 / 值域 / 零点）",
    },
    "preview_acc_extended": {
        LANG_EN: "Extended Analysis (Intercepts / Extrema / Monotonicity / Symmetry)",
        LANG_ZH: "拓展分析（截距 / 极值 / 单调性 / 对称性）",
    },
    "preview_acc_ints": {
        LANG_EN: "Intersection Points",
        LANG_ZH: "交点坐标",
    },
    "preview_acc_tasks": {
        LANG_EN: "Evaluation Tasks (Predictions / Rate of Change)",
        LANG_ZH: "评估任务（函数值预测 / 变化率）",
    },
    "preview_acc_meta": {
        LANG_EN: "Image & Grid Metadata",
        LANG_ZH: "图像与网格元信息",
    },

    "no_image_selected": {
        LANG_EN: "No image selected",
        LANG_ZH: "未选择图像",
    },
    "no_intersections": {
        LANG_EN: "No intersections",
        LANG_ZH: "无交点",
    },
    "image_not_found": {
        LANG_EN: "Image not found",
        LANG_ZH: "未找到图像",
    },
    "no_images_dataset": {
        LANG_EN: "No images in dataset",
        LANG_ZH: "数据集中无图像",
    },
    "language_label": {
        LANG_EN: "Language / 语言",
        LANG_ZH: "语言 / Language",
    },

    # =====================================================================
    # Error messages (evaluator adapters & pipeline)
    # =====================================================================
    "err_no_api_key": {
        LANG_EN: "API key not provided. Please set your API key.",
        LANG_ZH: "未提供 API Key，请先设置 API Key。",
    },
    "err_image_load_failed": {
        LANG_EN: "Failed to load image: {path}",
        LANG_ZH: "图像加载失败: {path}",
    },
    "err_chart_extraction_failed": {
        LANG_EN: "Chart extraction failed",
        LANG_ZH: "Chart 提取失败",
    },
    "err_empty_response": {
        LANG_EN: "Empty response from model",
        LANG_ZH: "模型返回空响应",
    },
    "err_recognition_failed": {
        LANG_EN: "Recognition failed: {detail}",
        LANG_ZH: "识别失败: {detail}",
    },
    "err_ocr_failed": {
        LANG_EN: "OCR call failed: {detail}",
        LANG_ZH: "OCR 调用失败: {detail}",
    },
    "err_ocr_no_line_data": {
        LANG_EN: "OCR result is invalid or missing line_data",
        LANG_ZH: "OCR 结果无效或未包含 line_data",
    },
    "err_gemini_failed": {
        LANG_EN: "Gemini recognition failed: {detail}",
        LANG_ZH: "Gemini 识别失败: {detail}",
    },

    # =====================================================================
    # Console log prefixes (for structured logging)
    # =====================================================================
    "log_chart_extract": {
        LANG_EN: "[Chart Extract]",
        LANG_ZH: "[Chart提取]",
    },
    "log_ai_enhance": {
        LANG_EN: "[AI Enhance]",
        LANG_ZH: "[AI增强]",
    },
    "log_model_manager": {
        LANG_EN: "[ModelManager]",
        LANG_ZH: "[ModelManager]",
    },

    # =====================================================================
    # Coordinate Recognizer UI  (prefix: rec_)
    # =====================================================================
    "rec_title": {
        LANG_EN: "# GeoSym Coordinate Recognition System",
        LANG_ZH: "# GeoSym 坐标识别系统",
    },
    "rec_desc": {
        LANG_EN: "**Five-stage pipeline**: S0 Chart Extraction → S0.5 AI Enhancement(opt) → S1 Preprocessing → S2 Coord Recognition → S3 Inverse Transform",
        LANG_ZH: "**五阶段处理流程**: S0 Chart提取 → S0.5 AI增强(可选) → S1 图像预处理 → S2 坐标识别 → S3 坐标逆变换",
    },
    # Stage 0
    "s0_title": {LANG_EN: "## S0: Problem Image Processing", LANG_ZH: "## S0: 题目图片处理"},
    "s0_desc": {
        LANG_EN: "Upload a problem image with a coordinate system; the system will auto-detect and extract the chart area",
        LANG_ZH: "上传包含坐标系的题目图片，系统将自动识别并提取坐标系区域",
    },
    "s0_upload": {LANG_EN: "Upload Problem Image", LANG_ZH: "上传题目图片"},
    "s0_advanced": {LANG_EN: "Advanced Parameters", LANG_ZH: "高级参数"},
    "s0_h_tolerance": {LANG_EN: "Horizontal Tolerance (X-axis)", LANG_ZH: "水平线容差（X轴标签）"},
    "s0_v_tolerance": {LANG_EN: "Vertical Tolerance (Y-axis)", LANG_ZH: "垂直线容差（Y轴标签）"},
    "s0_interval": {LANG_EN: "Interval Percentile", LANG_ZH: "间隔分位数"},
    "s0_gemini": {LANG_EN: "Enable Gemini Correction (optional, improves accuracy)", LANG_ZH: "启用Gemini校正（可选，提升准确率）"},
    "s0_btn": {LANG_EN: "Extract Chart", LANG_ZH: "提取坐标系"},
    "s0_tab_result": {LANG_EN: "Extraction Result", LANG_ZH: "提取结果"},
    "s0_tab_debug": {LANG_EN: "Visual Debug", LANG_ZH: "可视化调试"},
    "s0_tab_ocr_info": {LANG_EN: "OCR Details", LANG_ZH: "详细OCR信息"},
    "s0_tab_ocr_raw": {LANG_EN: "OCR Raw Data", LANG_ZH: "OCR原始数据"},
    "s0_tab_ocr_vis": {LANG_EN: "OCR Visualization", LANG_ZH: "OCR可视化"},
    "s0_tab_gemini": {LANG_EN: "Gemini Recognition", LANG_ZH: "Gemini识别"},
    "s0_chart_img": {LANG_EN: "Extracted Chart Image", LANG_ZH: "提取的坐标系图片"},
    "s0_info": {LANG_EN: "Extraction Info", LANG_ZH: "提取信息"},
    "s0_debug_vis": {LANG_EN: "Debug Visualization", LANG_ZH: "调试可视化"},
    "s0_ocr_debug_details": {LANG_EN: "OCR Debug Details", LANG_ZH: "OCR 调试详情"},
    "s0_ocr_json": {LANG_EN: "OCR JSON", LANG_ZH: "OCR JSON"},
    "s0_ocr_vis_label": {LANG_EN: "OCR Visualization", LANG_ZH: "OCR 可视化"},
    "s0_gemini_annotated": {LANG_EN: "Gemini Annotated", LANG_ZH: "Gemini 标注图"},
    "s0_ocr_vs_gemini": {LANG_EN: "### OCR vs Gemini", LANG_ZH: "### OCR vs Gemini"},
    # Stage 0.5
    "s05_title": {LANG_EN: "## S0.5: AI Image Enhancement (Optional)", LANG_ZH: "## S0.5: AI图片增强（可选）"},
    "s05_desc": {
        LANG_EN: "Use AI models to preprocess/enhance the extracted chart image",
        LANG_ZH: "使用AI模型对提取的坐标系图片进行预处理/增强",
    },
    "s05_enable": {LANG_EN: "Enable AI Processing", LANG_ZH: "启用AI处理"},
    "s05_model": {LANG_EN: "Select AI Model", LANG_ZH: "选择AI模型"},
    "s05_prompt": {LANG_EN: "Processing Prompt", LANG_ZH: "处理Prompt"},
    "s05_prompt_placeholder": {LANG_EN: "Enter processing prompt...", LANG_ZH: "输入处理 Prompt..."},
    "s05_btn": {LANG_EN: "Run AI Processing", LANG_ZH: "执行AI处理"},
    "s05_original": {LANG_EN: "Original Chart Image", LANG_ZH: "原始Chart图片"},
    "s05_processed": {LANG_EN: "After AI Processing", LANG_ZH: "AI处理后"},
    "s05_info": {LANG_EN: "Processing Info", LANG_ZH: "处理信息"},
    # Stage 1
    "s1_title": {LANG_EN: "## S1: Image Preprocessing", LANG_ZH: "## S1: 图像预处理"},
    "s1_desc": {
        LANG_EN: "Tilt correction and grid line detection on the extracted chart",
        LANG_ZH: "对提取的坐标系进行倾斜校正和网格线检测",
    },
    "s1_use_chart": {LANG_EN: "Use Stage 0/0.5 image", LANG_ZH: "使用阶段0/0.5的图片"},
    "s1_manual": {LANG_EN: "Or upload chart image manually", LANG_ZH: "或手动上传坐标系图片"},
    "s1_steps": {LANG_EN: "### Processing Steps", LANG_ZH: "### 处理步骤"},
    "s1_curve_erase": {LANG_EN: "Curve Detection (preprocessing)", LANG_ZH: "曲线检测（预处理）"},
    "s1_dashed": {LANG_EN: "Dashed Line Connection", LANG_ZH: "虚线连接"},
    "s1_perspective": {LANG_EN: "Perspective Correction", LANG_ZH: "透视校正"},
    "s1_rotation": {LANG_EN: "Tilt Correction", LANG_ZH: "倾斜校正"},
    "s1_curve_params": {LANG_EN: "Curve Detection Parameters", LANG_ZH: "曲线检测参数"},
    "s1_dashed_params": {LANG_EN: "Dashed Connection Parameters", LANG_ZH: "虚线连接参数"},
    "s1_rot_params": {LANG_EN: "Tilt Correction Parameters", LANG_ZH: "倾斜校正参数"},
    "s1_grid_params": {LANG_EN: "Grid Detection Parameters", LANG_ZH: "网格检测参数"},
    "s1_merge_params": {LANG_EN: "Line Merge Parameters", LANG_ZH: "线段合并参数"},
    "s1_btn": {LANG_EN: "Start Processing", LANG_ZH: "开始处理"},
    "s1_tab_final": {LANG_EN: "Final Result", LANG_ZH: "最终结果"},
    "s1_tab_edges": {LANG_EN: "Edge Detection", LANG_ZH: "边缘检测"},
    "s1_tab_lines": {LANG_EN: "Raw Line Detection", LANG_ZH: "原始线条检测"},
    "s1_tab_dashed": {LANG_EN: "Dashed Connection Result", LANG_ZH: "虚线连接结果"},
    "s1_tab_merged": {LANG_EN: "Merged Grid Lines", LANG_ZH: "合并后网格线"},
    "s1_info": {LANG_EN: "Processing Info", LANG_ZH: "处理信息"},
    "s1_feather_size": {LANG_EN: "Feather Size", LANG_ZH: "羽化大小"},
    "s1_gap_threshold": {LANG_EN: "Gap Threshold (px)", LANG_ZH: "间隙阈值（像素）"},
    "s1_collinear_threshold": {LANG_EN: "Collinear Threshold (px)", LANG_ZH: "共线阈值（像素）"},
    "s1_hough_threshold": {LANG_EN: "Hough Threshold", LANG_ZH: "Hough 阈值"},
    "s1_min_line_length": {LANG_EN: "Min Line Length", LANG_ZH: "最小线段长度"},
    "s1_max_line_gap": {LANG_EN: "Max Line Gap", LANG_ZH: "最大线段间隙"},
    "s1_angle_threshold": {LANG_EN: "Angle Threshold", LANG_ZH: "角度阈值"},
    "s1_merge_dist": {LANG_EN: "Merge Distance Threshold (px)", LANG_ZH: "合并距离阈值（像素）"},
    "s1_processed_img": {LANG_EN: "Processed Image", LANG_ZH: "处理后图像"},
    "s1_edge_detect": {LANG_EN: "Edge Detection", LANG_ZH: "边缘检测"},
    "s1_detected_lines": {LANG_EN: "Detected Lines", LANG_ZH: "检测到的线条"},
    "s1_dashed_conn": {LANG_EN: "Dashed Connection", LANG_ZH: "虚线连接"},
    "s1_merged_grid": {LANG_EN: "Merged Grid Lines", LANG_ZH: "合并后网格线"},
    # Stage 2
    "s2_title": {LANG_EN: "## S2: Coordinate Recognition", LANG_ZH: "## S2: 坐标识别"},
    "s2_desc": {
        LANG_EN: "Detect curve-grid intersection coordinates based on Stage 1 results",
        LANG_ZH: "基于阶段1结果，识别曲线与坐标轴/网格的交点坐标",
    },
    "s2_params": {LANG_EN: "### Parameter Settings", LANG_ZH: "### 参数设置"},
    "s2_tolerance": {LANG_EN: "Intersection Tolerance (px)", LANG_ZH: "交点容差（像素）"},
    "s2_erase": {LANG_EN: "Erase Line Thickness (px)", LANG_ZH: "擦除线条粗细（像素）"},
    "s2_curve_extract": {LANG_EN: "Curve Extraction Parameters", LANG_ZH: "曲线提取参数"},
    "s2_sat_thresh": {LANG_EN: "Color Saturation Threshold", LANG_ZH: "颜色饱和度阈值"},
    "s2_val_thresh": {LANG_EN: "Color Value Threshold", LANG_ZH: "颜色亮度阈值"},
    "s2_morph_enable": {LANG_EN: "Enable Fracture Connection", LANG_ZH: "启用连接断裂部分"},
    "s2_morph_kernel": {LANG_EN: "Connection Kernel Size", LANG_ZH: "连接核大小"},
    "s2_morph_iter": {LANG_EN: "Connection Iterations", LANG_ZH: "连接迭代次数"},
    "s2_filter_params": {LANG_EN: "Intersection Filter Parameters", LANG_ZH: "交点过滤参数"},
    "s2_integer_thresh": {LANG_EN: "Integer Approximation Threshold", LANG_ZH: "整数近似阈值"},
    "s2_btn": {LANG_EN: "Start Recognition", LANG_ZH: "开始识别坐标"},
    "s2_info": {LANG_EN: "Recognition Info", LANG_ZH: "识别信息"},
    "s2_tab_intermediate": {LANG_EN: "Intermediate Steps", LANG_ZH: "中间过程"},
    "s2_tab_curve": {LANG_EN: "Extracted Curve", LANG_ZH: "提取的曲线"},
    "s2_tab_grid_coords": {LANG_EN: "Grid Coordinates", LANG_ZH: "网格线坐标"},
    "s2_tab_annotated": {LANG_EN: "Annotated Image", LANG_ZH: "标注图"},
    "s2_tab_raw_table": {LANG_EN: "Raw Intersections (before filter)", LANG_ZH: "原始交点表（过滤前）"},
    "s2_tab_filtered_table": {LANG_EN: "Key Intersections (after filter)", LANG_ZH: "关键交点表（过滤后）"},
    "s2_tab_ruler": {LANG_EN: "Ruler Verification", LANG_ZH: "标尺交叉验证"},
    "s2_ruler_info": {LANG_EN: "Ruler Cross-Validation Details", LANG_ZH: "标尺交叉验证详情"},
    "s2_ruler_formula": {LANG_EN: "Regression Formula", LANG_ZH: "回归公式"},
    "s2_ruler_table": {LANG_EN: "Grid vs Ruler Comparison", LANG_ZH: "网格吸附 vs 连续插值 对比"},
    "s2_binarized": {LANG_EN: "1. Binarized", LANG_ZH: "1. 二值化"},
    "s2_skeleton": {LANG_EN: "2. Skeleton", LANG_ZH: "2. 骨架化"},
    "s2_filtered": {LANG_EN: "3. Filtered", LANG_ZH: "3. 过滤"},
    "s2_connected": {LANG_EN: "4. Connected", LANG_ZH: "4. 连通"},
    "s2_extracted_curve": {LANG_EN: "Extracted Curve", LANG_ZH: "提取的曲线"},
    "s2_grid_coord_map": {LANG_EN: "Grid Coordinate Map", LANG_ZH: "网格坐标图"},
    "s2_annotated_img": {LANG_EN: "Annotated Image", LANG_ZH: "标注图像"},
    "s2_raw_intersections": {LANG_EN: "Raw Intersections", LANG_ZH: "原始交点"},
    "s2_filtered_intersections": {LANG_EN: "Filtered Intersections", LANG_ZH: "过滤后交点"},
    # Stage 3
    "s3_title": {LANG_EN: "## S3: Inverse Transform & Annotation", LANG_ZH: "## S3: 坐标逆变换和原图标注"},
    "s3_desc": {
        LANG_EN: "Map Stage 2 intersections back to original image and draw annotations",
        LANG_ZH: "将阶段2识别的交点坐标映射回原始图像，并在原图上绘制标注",
    },
    "s3_btn": {LANG_EN: "Run Inverse Transform", LANG_ZH: "执行坐标逆变换"},
    "s3_info": {LANG_EN: "Processing Info", LANG_ZH: "处理信息"},
    "s3_tab_ai": {LANG_EN: "Annotated (AI Enhanced)", LANG_ZH: "原图标注（AI增强）"},
    "s3_tab_raw": {LANG_EN: "Annotated (Original)", LANG_ZH: "原图标注（原始）"},
    "s3_tab_table": {LANG_EN: "Coordinate Comparison Table", LANG_ZH: "坐标对照表"},
    "s3_ai_annotation": {LANG_EN: "AI Enhanced Annotation", LANG_ZH: "AI增强标注图"},
    "s3_orig_annotation": {LANG_EN: "Original Annotation", LANG_ZH: "原始标注图"},
    "s3_coord_comparison": {LANG_EN: "Coordinate Comparison", LANG_ZH: "坐标对照"},
    "s3_facts_panel": {LANG_EN: "Recognized Visual Facts", LANG_ZH: "识别到的视觉事实"},
    "s3_facts_hint": {
        LANG_EN: "Run Stage 3 to view recognized intercepts, extrema, and visible-range monotonicity.",
        LANG_ZH: "执行 Stage 3 后，这里将显示识别到的截距、极值点和可见范围单调性。",
    },

    # =====================================================================
    # Recognizer handler messages  (prefix: rec_msg_ / rec_err_)
    # =====================================================================
    "rec_err_upload_first": {
        LANG_EN: "Please upload a problem image first.",
        LANG_ZH: "请先上传题目图片。",
    },
    "rec_err_chart_failed": {
        LANG_EN: "Chart extraction failed: {detail}",
        LANG_ZH: "Chart 提取失败: {detail}",
    },
    "rec_err_chart_empty": {
        LANG_EN: "Chart image is empty",
        LANG_ZH: "Chart 图像为空",
    },
    "rec_msg_chart_success": {
        LANG_EN: "Chart extracted successfully!",
        LANG_ZH: "坐标系提取成功！",
    },
    "rec_err_stage0_first": {
        LANG_EN: "Please complete Stage 0 first.",
        LANG_ZH: "请先完成阶段 0。",
    },
    "rec_msg_ai_skipped": {
        LANG_EN: "Skipped AI processing, using original image.",
        LANG_ZH: "跳过 AI 处理，使用原始图像。",
    },
    "rec_err_enter_prompt": {
        LANG_EN: "Please enter a processing prompt.",
        LANG_ZH: "请输入处理 Prompt。",
    },
    "rec_err_stage1_first": {
        LANG_EN: "Please complete Stage 1 first.",
        LANG_ZH: "请先完成阶段 1。",
    },
    "rec_err_extract_or_upload": {
        LANG_EN: "Please extract chart or upload an image.",
        LANG_ZH: "请先提取坐标系或上传图像。",
    },
    "rec_err_missing_labels": {
        LANG_EN: "Missing tick label info. Please ensure Stage 0 extracted the chart and Stage 1 used Stage 0 results.",
        LANG_ZH: "缺少刻度标签信息。请确保阶段 0 已提取坐标系，且阶段 1 使用了阶段 0 的结果。",
    },
    "rec_msg_recognition_success": {
        LANG_EN: "Recognition successful!",
        LANG_ZH: "识别成功！",
    },
    "rec_err_stage2_first": {
        LANG_EN: "Please complete Stage 2 first.",
        LANG_ZH: "请先完成阶段 2。",
    },
    "rec_err_missing_orig": {
        LANG_EN: "Missing original chart image. Please ensure Stage 1 used Stage 0 results.",
        LANG_ZH: "缺少原始坐标系图像。请确保阶段 1 使用了阶段 0 的结果。",
    },
    "rec_err_no_intersections_s2": {
        LANG_EN: "No intersections found in Stage 2.",
        LANG_ZH: "阶段 2 未发现交点。",
    },
    "rec_msg_stage3_complete": {
        LANG_EN: "Stage 3 Complete!",
        LANG_ZH: "阶段 3 完成！",
    },

    # =====================================================================
    # Evaluator adapter messages  (prefix: eval_)
    # =====================================================================
    "eval_gemini_key_missing": {
        LANG_EN: "Gemini API key not provided. Get one at https://aistudio.google.com/apikey",
        LANG_ZH: "未提供 Gemini API Key，请前往 https://aistudio.google.com/apikey 获取",
    },
    "eval_gemini_empty_response": {
        LANG_EN: "[Empty response from Gemini] {detail}",
        LANG_ZH: "[Gemini 返回空响应] {detail}",
    },
    "eval_openai_key_missing": {
        LANG_EN: "OpenAI API key not provided. Please set api_key.",
        LANG_ZH: "未提供 OpenAI API Key，请设置 api_key。",
    },
    "eval_glm_key_missing": {
        LANG_EN: "ZhipuAI API key not provided. Get one at https://bigmodel.cn/usercenter/proj-mgmt/apikeys",
        LANG_ZH: "未提供智谱 API Key，请前往 https://bigmodel.cn/usercenter/proj-mgmt/apikeys 获取",
    },
    "eval_qwen_key_missing": {
        LANG_EN: "Bailian API key not provided. Get one at https://bailian.console.aliyun.com/?tab=model#/api-key",
        LANG_ZH: "未提供百炼 API Key，请前往 https://bailian.console.aliyun.com/?tab=model#/api-key 获取",
    },
    "eval_recognizer_load_failed": {
        LANG_EN: "Failed to load image: {path}",
        LANG_ZH: "加载图像失败: {path}",
    },
    "eval_recognizer_extract_failed": {
        LANG_EN: "Chart extraction failed",
        LANG_ZH: "Chart 提取失败",
    },

    # =====================================================================
    # Connection Test  (prefix: conn_)
    # =====================================================================
    "tab_conn_test": {
        LANG_EN: "🔌 Connection Test",
        LANG_ZH: "🔌 连接测试",
    },
    "conn_test_title": {
        LANG_EN: "### Model Connection Test\nValidate API key configuration and network connectivity for each LLM provider.\nA minimal vision request (64×64 white image) is sent to confirm the model is reachable.",
        LANG_ZH: "### 模型连接测试\n验证各 LLM 供应商的 API Key 配置与网络连通性。\n发送一个最小视觉请求（64×64 白色图片）来确认模型可达。",
    },
    "conn_overview_title": {
        LANG_EN: "#### Configured Providers & Models",
        LANG_ZH: "#### 已配置的供应商与模型",
    },
    "conn_provider": {
        LANG_EN: "Provider",
        LANG_ZH: "供应商",
    },
    "conn_model": {
        LANG_EN: "Model",
        LANG_ZH: "模型",
    },
    "conn_test_btn": {
        LANG_EN: "🔍 Test Connection",
        LANG_ZH: "🔍 测试连接",
    },
    "conn_result_label": {
        LANG_EN: "Test Results",
        LANG_ZH: "测试结果",
    },

    "reason_clear_confirm": {
        LANG_EN: "✅ Cleared all cached data for current image ({files} files removed).",
        LANG_ZH: "✅ 已清空当前图像的所有缓存数据（共删除 {files} 个文件）。",
    },
    "reason_clear_empty": {
        LANG_EN: "No cached data found for this image.",
        LANG_ZH: "未找到当前图像的缓存数据。",
    },
    "reason_clear_all_confirm": {
        LANG_EN: "⚠️ All reasoning cache has been deleted ({count} files removed).",
        LANG_ZH: "⚠️ 已清空全部推理缓存（共删除 {count} 个文件）。",
    },
    "reason_clear_all_empty": {
        LANG_EN: "No reasoning cache found.",
        LANG_ZH: "未找到任何推理缓存数据。",
    },
    "reason_clear_no_image": {
        LANG_EN: "⚠️ Please select an image first.",
        LANG_ZH: "⚠️ 请先选择一张图像。",
    },
}


_LOG_TAGS: Dict[str, Dict[str, str]] = {
    "AI增强": {LANG_EN: "AI Enhance", LANG_ZH: "AI增强"},
    "API异常": {LANG_EN: "API Error", LANG_ZH: "API异常"},
    "Chart提取": {LANG_EN: "ChartExtract", LANG_ZH: "Chart提取"},
    "Gemini识别": {LANG_EN: "GeminiRecog", LANG_ZH: "Gemini识别"},
    "Gemini输入": {LANG_EN: "GeminiInput", LANG_ZH: "Gemini输入"},
    "OCR标签提取": {LANG_EN: "OCRLabelExtract", LANG_ZH: "OCR标签提取"},
    "交点过滤": {LANG_EN: "IntersFilter", LANG_ZH: "交点过滤"},
    "位置聚类": {LANG_EN: "PosClustering", LANG_ZH: "位置聚类"},
    "刻度标签变换": {LANG_EN: "TickTransform", LANG_ZH: "刻度标签变换"},
    "参数": {LANG_EN: "Params", LANG_ZH: "参数"},
    "可视化": {LANG_EN: "Visualization", LANG_ZH: "可视化"},
    "图像加载": {LANG_EN: "ImageLoad", LANG_ZH: "图像加载"},
    "图像旋转": {LANG_EN: "ImageRotate", LANG_ZH: "图像旋转"},
    "图像编码": {LANG_EN: "ImageEncode", LANG_ZH: "图像编码"},
    "图像裁剪": {LANG_EN: "ImageCrop", LANG_ZH: "图像裁剪"},
    "坐标变换": {LANG_EN: "CoordTransform", LANG_ZH: "坐标变换"},
    "坐标映射": {LANG_EN: "CoordMapping", LANG_ZH: "坐标映射"},
    "坐标识别": {LANG_EN: "CoordRecog", LANG_ZH: "坐标识别"},
    "坐标轴校准": {LANG_EN: "AxisCalibrate", LANG_ZH: "坐标轴校准"},
    "完整流程": {LANG_EN: "FullPipeline", LANG_ZH: "完整流程"},
    "尺寸调整": {LANG_EN: "Resize", LANG_ZH: "尺寸调整"},
    "开关": {LANG_EN: "Toggle", LANG_ZH: "开关"},
    "数值过滤": {LANG_EN: "NumFilter", LANG_ZH: "数值过滤"},
    "旋转校正": {LANG_EN: "RotCorrection", LANG_ZH: "旋转校正"},
    "曲线提取": {LANG_EN: "CurveExtract", LANG_ZH: "曲线提取"},
    "曲线擦除": {LANG_EN: "CurveErase", LANG_ZH: "曲线擦除"},
    "曲线检测": {LANG_EN: "CurveDetect", LANG_ZH: "曲线检测"},
    "曲线过滤": {LANG_EN: "CurveFilter", LANG_ZH: "曲线过滤"},
    "曲线连接": {LANG_EN: "CurveConnect", LANG_ZH: "曲线连接"},
    "最终结果": {LANG_EN: "FinalResult", LANG_ZH: "最终结果"},
    "线段合并": {LANG_EN: "LineMerge", LANG_ZH: "线段合并"},
    "绘图工具": {LANG_EN: "DrawUtil", LANG_ZH: "绘图工具"},
    "网格检测": {LANG_EN: "GridDetect", LANG_ZH: "网格检测"},
    "虚线连接": {LANG_EN: "DashedConn", LANG_ZH: "虚线连接"},
    "裁剪计算": {LANG_EN: "CropCalc", LANG_ZH: "裁剪计算"},
    "角度计算": {LANG_EN: "AngleCalc", LANG_ZH: "角度计算"},
    "角点检测": {LANG_EN: "CornerDetect", LANG_ZH: "角点检测"},
    "调试": {LANG_EN: "Debug", LANG_ZH: "调试"},
    "轴推断": {LANG_EN: "AxisInfer", LANG_ZH: "轴推断"},
    "逆变换": {LANG_EN: "InvTransform", LANG_ZH: "逆变换"},
    "透视校正": {LANG_EN: "PerspCorrect", LANG_ZH: "透视校正"},
    "阶段3": {LANG_EN: "Stage3", LANG_ZH: "阶段3"},
    "预处理": {LANG_EN: "Preprocess", LANG_ZH: "预处理"},
    "步骤1": {LANG_EN: "Step1", LANG_ZH: "步骤1"},
    "步骤1.5": {LANG_EN: "Step1.5", LANG_ZH: "步骤1.5"},
    "步骤1.6": {LANG_EN: "Step1.6", LANG_ZH: "步骤1.6"},
    "步骤1总结": {LANG_EN: "Step1Summary", LANG_ZH: "步骤1总结"},
    "步骤2": {LANG_EN: "Step2", LANG_ZH: "步骤2"},
    "步骤3": {LANG_EN: "Step3", LANG_ZH: "步骤3"},
    "步骤3总结": {LANG_EN: "Step3Summary", LANG_ZH: "步骤3总结"},
    "步骤4": {LANG_EN: "Step4", LANG_ZH: "步骤4"},
    "步骤4.1": {LANG_EN: "Step4.1", LANG_ZH: "步骤4.1"},
    "步骤4.2": {LANG_EN: "Step4.2", LANG_ZH: "步骤4.2"},
    "步骤4.3": {LANG_EN: "Step4.3", LANG_ZH: "步骤4.3"},
    "步骤4.3总结": {LANG_EN: "Step4.3Sum", LANG_ZH: "步骤4.3总结"},
    "步骤4.4": {LANG_EN: "Step4.4", LANG_ZH: "步骤4.4"},
    "步骤4总结": {LANG_EN: "Step4Summary", LANG_ZH: "步骤4总结"},
    "ModelManager": {LANG_EN: "ModelManager", LANG_ZH: "ModelManager"},
    "ImageEnhancer": {LANG_EN: "ImageEnhancer", LANG_ZH: "ImageEnhancer"},
}


def log_tag(zh_tag: str) -> str:
    """Translate a log bracket tag, e.g. ``log_tag("Chart提取")`` → ``"ChartExtract"``."""
    entry = _LOG_TAGS.get(zh_tag)
    if entry:
        return entry.get(_current_lang, zh_tag)
    return zh_tag


_current_lang: str = LANG_ZH


def set_lang(lang: str) -> None:
    """Set the global UI language (called on language switch)."""
    global _current_lang
    _current_lang = lang


def get_lang() -> str:
    """Return the current global UI language."""
    return _current_lang


def t(key: str, lang: str | None = None) -> str:
    """Get translated text for the given key and language.

    If *lang* is ``None``, the current global UI language is used.
    """
    if lang is None:
        lang = _current_lang
    return TEXTS.get(key, {}).get(lang, key)
