"""
Enhanced M0 对比测试脚本

对选定样本运行增强 M0 (OCR标尺 + LLM定位 + 数值计算)，
与现有纯 LLM M0 的 baseline 报告进行对比。

用法:
  python scripts/test_enhanced_m0.py [sample_id ...]

默认测试: fred_001 engisphere_001 engisphere_008
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from presenter.ruler import build_ruler, format_ruler_for_prompt
from presenter.measure import measure_points, compute_derived_quantities, pixel_to_real
from presenter.verify import verify_measurements, fuse_evidence
from presenter.service import (
    _load_sample_spec, _ensure_adapter, _load_prompt,
    _call_json_stage, _run_llm_m0_extraction, REPORT_ROOT,
)


def _load_baseline_m0(sample_id: str) -> dict | None:
    """加载现有 baseline 的 M0 结果"""
    report_dir = REPORT_ROOT / sample_id
    if not report_dir.exists():
        return None
    for run_dir in sorted(report_dir.iterdir()):
        m0_path = run_dir / "geosym_m0.json"
        if m0_path.exists():
            return json.loads(m0_path.read_text(encoding="utf-8"))
    return None


def _load_baseline_final(sample_id: str) -> dict | None:
    """加载现有 baseline 的最终答案"""
    report_dir = REPORT_ROOT / sample_id
    if not report_dir.exists():
        return None
    for run_dir in sorted(report_dir.iterdir()):
        final_path = run_dir / "geosym_final.json"
        if final_path.exists():
            return json.loads(final_path.read_text(encoding="utf-8"))
    return None


def _extract_numeric(text: str) -> list[float]:
    """从文本中提取所有数字"""
    return [float(m) for m in re.findall(r"[-+]?\d+\.?\d*", text)]


def _check_gold_answer(result_facts: list[str], gold: dict) -> dict:
    """检查结果是否匹配 gold answer"""
    joined = " ".join(result_facts).lower()

    # 检查必需短语
    phrase_hits = []
    for group in gold.get("required_any_phrase_groups", []):
        hit = any(phrase.lower() in joined for phrase in group)
        phrase_hits.append({"phrases": group, "hit": hit})

    # 检查数字目标
    all_nums = _extract_numeric(joined)
    numeric_hits = []
    for target in gold.get("required_numeric_targets", []):
        tv = target["value"]
        tol = target["tolerance"]
        hit = any(abs(n - tv) <= tol for n in all_nums)
        numeric_hits.append({"target": tv, "tolerance": tol, "hit": hit})

    all_pass = all(p["hit"] for p in phrase_hits) and all(n["hit"] for n in numeric_hits)
    return {"pass": all_pass, "phrases": phrase_hits, "numerics": numeric_hits}


def run_phase1_only(sample_id: str):
    """仅运行 Phase 1 (Build Ruler) 测试 OCR 标尺构建"""
    sample = _load_sample_spec(sample_id)
    image_path = str((_root / sample["image_path"]).resolve())

    print(f"\n{'='*70}")
    print(f"  Phase 1 Test: {sample_id}")
    print(f"{'='*70}")
    print(f"  Image: {image_path}")

    ruler = build_ruler(image_path)

    print(f"\n  Ruler success: {ruler.success}")
    print(f"  Confidence: {ruler.confidence}")

    if ruler.x_axis:
        ax = ruler.x_axis
        print(f"\n  X-axis: scale={ax.scale_type}")
        print(f"    Ticks: {len(ax.tick_pixels)} detected")
        for px, val in zip(ax.tick_pixels[:5], ax.tick_values[:5]):
            print(f"      pixel_x={px:.0f} → {val}")
        if len(ax.tick_pixels) > 5:
            print(f"      ... and {len(ax.tick_pixels)-5} more")
        print(f"    Linear fit: value = {ax.scale:.6f} × pixel + {ax.offset:.2f}")

    if ruler.y_axis:
        ax = ruler.y_axis
        print(f"\n  Y-axis: scale={ax.scale_type}")
        print(f"    Ticks: {len(ax.tick_pixels)} detected")
        for px, val in zip(ax.tick_pixels, ax.tick_values):
            print(f"      pixel_y={px:.0f} → {val}")
        print(f"    Linear fit: value = {ax.scale:.6f} × pixel + {ax.offset:.2f}")

    if ruler.success:
        print(f"\n  Plot region: ({ruler.plot_region.x0:.0f}, {ruler.plot_region.y0:.0f}) → "
              f"({ruler.plot_region.x1:.0f}, {ruler.plot_region.y1:.0f})")

        # 验算: 用标尺反算刻度位置，检查误差
        if ruler.y_axis and ruler.y_axis.scale_type == "linear":
            print(f"\n  Y-axis 反算验证:")
            for px, val in zip(ruler.y_axis.tick_pixels, ruler.y_axis.tick_values):
                computed = ruler.y_axis.pixel_to_value(px)
                err = abs(computed - float(val))
                mark = "✓" if err < 0.5 else "✗"
                print(f"    {mark} pixel_y={px:.0f} → computed={computed:.2f}, expected={val}, error={err:.3f}")

    return ruler


def run_full_enhanced_m0(sample_id: str, model_name: str = "GPT-5.4"):
    """运行完整的增强 M0 管线并与 baseline 对比"""
    sample = _load_sample_spec(sample_id)
    image_path = str((_root / sample["image_path"]).resolve())
    gold = sample.get("answer_validation", {})

    print(f"\n{'='*70}")
    print(f"  Full Enhanced M0 Test: {sample_id}")
    print(f"  Model: {model_name}")
    print(f"  Question: {sample.get('question', '')}")
    print(f"{'='*70}")

    # Phase 1: Build Ruler
    print("\n[Phase 1] Building ruler...")
    ruler = build_ruler(image_path)
    if not ruler.success:
        print("  ✗ Ruler failed, cannot run enhanced M0")
        return None

    print(f"  ✓ Ruler built: confidence={ruler.confidence}")

    # Phase 2: LLM Locate
    print("\n[Phase 2] LLM locating data points...")
    adapter = _ensure_adapter(model_name)
    ruler_context = format_ruler_for_prompt(ruler)
    locate_prompt = _load_prompt("m0_locate")
    locate_prompt = locate_prompt.replace("{ruler_context}", ruler_context)
    locate_prompt = locate_prompt.replace("{question}", sample.get("question", ""))

    locate_fallback = {
        "located_points": [],
        "semantic_observations": {
            "chart_type": "unknown", "series_count": 1,
            "overall_trend": "", "notable_features": [],
        },
    }
    locate_parsed, locate_raw, fallback_used = _call_json_stage(
        adapter, image_path, locate_prompt, locate_fallback
    )
    located = locate_parsed.get("located_points", [])
    semantic = locate_parsed.get("semantic_observations", {})
    print(f"  ✓ Located {len(located)} points (fallback={fallback_used})")

    for pt in located:
        print(f"    {pt.get('role', '?'):12s} pixel=({pt.get('pixel_x', 0):.0f}, {pt.get('pixel_y', 0):.0f})  "
              f"estimate={pt.get('estimated_value', '?')}")

    # Phase 3: Measure
    print("\n[Phase 3] Numerical measurement...")
    measured = measure_points(ruler, located)
    derived = compute_derived_quantities(measured, sample.get("question", ""))

    for pt in measured:
        print(f"    {pt.role:12s} ({pt.pixel_x:.0f},{pt.pixel_y:.0f}) → "
              f"x={pt.computed_x}, y={pt.computed_y}  [conf={pt.measurement_confidence}]"
              f"  (LLM estimated: {pt.llm_estimate})")

    for df in derived:
        print(f"    DERIVED: {df.description}: {df.formula}")

    # Phase 4: Verify & Fuse
    print("\n[Phase 4] Verifying and fusing...")
    verified = verify_measurements(ruler, measured)
    fused = fuse_evidence(ruler, verified, derived, semantic)

    print(f"  ✓ {len(verified)} points verified")
    print(f"  ✓ {len(fused['candidate_facts'])} candidate facts")
    print(f"  ✓ {len(fused['verified_facts'])} verified facts")
    print(f"  ✓ {len(fused['useful_facts'])} useful facts")

    # === 对比 ===
    print(f"\n{'─'*70}")
    print(f"  COMPARISON: Enhanced M0 vs Baseline (pure LLM M0)")
    print(f"{'─'*70}")

    # Gold answer check
    if gold:
        enhanced_check = _check_gold_answer(
            fused["candidate_facts"] + fused["useful_facts"], gold
        )
        print(f"\n  Enhanced M0 vs Gold Answer: {'✓ PASS' if enhanced_check['pass'] else '✗ FAIL'}")
        for p in enhanced_check["phrases"]:
            print(f"    Phrase match {p['phrases'][:2]}: {'✓' if p['hit'] else '✗'}")
        for n in enhanced_check["numerics"]:
            print(f"    Numeric target {n['target']} ± {n['tolerance']}: {'✓' if n['hit'] else '✗'}")

    # Load baseline
    baseline = _load_baseline_m0(sample_id)
    if baseline:
        baseline_parsed = baseline.get("parsed", {})
        baseline_facts = baseline_parsed.get("candidate_facts", []) + baseline_parsed.get("useful_facts", [])

        if gold:
            baseline_check = _check_gold_answer(baseline_facts, gold)
            print(f"\n  Baseline M0 vs Gold Answer:  {'✓ PASS' if baseline_check['pass'] else '✗ FAIL'}")
            for p in baseline_check["phrases"]:
                print(f"    Phrase match {p['phrases'][:2]}: {'✓' if p['hit'] else '✗'}")
            for n in baseline_check["numerics"]:
                print(f"    Numeric target {n['target']} ± {n['tolerance']}: {'✓' if n['hit'] else '✗'}")

        # 对比关键差异
        print(f"\n  Key differences:")
        print(f"    Baseline candidate_facts: {len(baseline_parsed.get('candidate_facts', []))}")
        print(f"    Enhanced candidate_facts: {len(fused['candidate_facts'])}")

        # 检查增强版是否有 [computed] 标记
        computed_count = sum(1 for f in fused["useful_facts"] if "[computed" in f)
        baseline_approx = sum(1 for f in baseline_facts if "approximately" in f.lower())
        print(f"    Baseline 'approximately' count: {baseline_approx}")
        print(f"    Enhanced '[computed]' count: {computed_count}")
        print(f"    Enhanced has computation_trace: {bool(fused.get('computation_trace'))}")

    # 输出 useful_facts 对比
    print(f"\n  Enhanced useful_facts:")
    for f in fused["useful_facts"]:
        print(f"    • {f}")

    if baseline:
        print(f"\n  Baseline useful_facts:")
        for f in baseline_parsed.get("useful_facts", []):
            print(f"    • {f}")

    return fused


def main():
    default_samples = ["fred_001", "engisphere_001", "engisphere_008"]
    samples = sys.argv[1:] if len(sys.argv) > 1 else default_samples

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         GeoSym Enhanced M0 — Comparative Test                      ║")
    print("║         OCR Ruler + LLM Locate + Numerical Measure                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Step 1: Phase 1 only (no LLM calls, fast)
    print("\n\n" + "="*70)
    print("  STAGE 1: Ruler Construction Test (no LLM, fast)")
    print("="*70)

    rulers = {}
    for sid in samples:
        try:
            rulers[sid] = run_phase1_only(sid)
        except Exception as e:
            print(f"\n  ✗ {sid}: {e}")

    # Step 2: Full pipeline (needs LLM API calls)
    print("\n\n" + "="*70)
    print("  STAGE 2: Full Enhanced M0 Pipeline (requires LLM API)")
    print("="*70)

    results = {}
    for sid in samples:
        if sid in rulers and rulers[sid].success:
            try:
                results[sid] = run_full_enhanced_m0(sid)
            except Exception as e:
                print(f"\n  ✗ {sid}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n  ⚠ Skipping {sid}: ruler not available")

    # Summary
    print("\n\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    for sid in samples:
        ruler_ok = sid in rulers and rulers[sid].success
        result_ok = sid in results and results[sid] is not None
        ruler_conf = rulers[sid].confidence if ruler_ok else "N/A"
        print(f"  {sid:25s}  ruler={ruler_conf:6s}  enhanced_m0={'✓' if result_ok else '✗'}")


if __name__ == "__main__":
    main()
