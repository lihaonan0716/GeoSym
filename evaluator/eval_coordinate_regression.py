"""
Recognizer M0 坐标映射回归测试脚本

用途: 升级 build_coordinate_mapping() 前后的 "before/after" 定量对比。

流程:
  1. 生成小规模合成 benchmark（或复用已有）
  2. 对每张图运行 RecognizerAdapter 完整管线
  3. 比对检测坐标 vs GT 坐标，计算 Jaccard / Precision / Recall / F1
  4. 保存 JSON 报告

用法:
  # 首次: 生成 benchmark + 跑 baseline
  python scripts/eval_coordinate_regression.py --generate --label baseline

  # 修改代码后: 只跑评估（复用已有 benchmark）
  python scripts/eval_coordinate_regression.py --label upgraded

  # 对比两份报告
  python scripts/eval_coordinate_regression.py --compare baseline.json upgraded.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

BENCH_DIR = _root / "outputs" / "regression_bench"
REPORT_DIR = _root / "outputs" / "regression_reports"

# 测试规模: 3 function types × 2 profiles × 3 images = 18 samples
TEST_FUNC_TYPES = ["linear", "quadratic", "cubic"]
TEST_PROFILES = ["clean_render", "digital_mild"]
IMAGES_PER_SUBSET = 3


def generate_benchmark():
    """生成小规模合成 benchmark"""
    from generator.benchmark import generate_benchmark as gen_bench

    print(f"Generating benchmark to {BENCH_DIR} ...")
    print(f"  func_types: {TEST_FUNC_TYPES}")
    print(f"  profiles: {TEST_PROFILES}")
    print(f"  images_per_subset: {IMAGES_PER_SUBSET}")

    from itertools import product
    subsets = list(product(TEST_FUNC_TYPES, TEST_PROFILES))

    result = gen_bench(
        output_dir=BENCH_DIR,
        images_per_subset=IMAGES_PER_SUBSET,
        base_seed=12345,
        subsets=subsets,
    )
    total = result.get("total_images", 0)
    print(f"  Generated {total} images")
    return BENCH_DIR


def _collect_samples(bench_dir: Path):
    """收集所有 (image_path, gt_intersections) 对

    Benchmark 目录结构:
      bench_dir/images/{subset}/{subset}_0000.png
      bench_dir/ground_truth/{subset}/{subset}_0000.json
    """
    samples = []
    images_dir = bench_dir / "images"
    gt_dir = bench_dir / "ground_truth"

    if not images_dir.exists() or not gt_dir.exists():
        return samples

    for subset_dir in sorted(images_dir.iterdir()):
        if not subset_dir.is_dir():
            continue
        subset_name = subset_dir.name
        gt_subset_dir = gt_dir / subset_name

        for img_file in sorted(subset_dir.glob("*.png")):
            sample_id = img_file.stem
            gt_path = gt_subset_dir / f"{sample_id}.json"
            if gt_path.exists():
                gt = json.loads(gt_path.read_text(encoding="utf-8"))
                intersections = gt.get("intersections", [])
                if intersections:
                    samples.append({
                        "image_path": str(img_file),
                        "gt_intersections": intersections,
                        "subset": subset_name,
                        "sample_id": sample_id,
                    })
    return samples


def run_evaluation(bench_dir: str | Path, label: str = "baseline"):
    """运行 Recognizer 管线并计算指标"""
    from evaluator.adapters.recognizer import RecognizerAdapter
    from evaluator.metrics.coordinate import compare_results, compute_metrics, aggregate_metrics

    bench_dir = Path(bench_dir)
    samples = _collect_samples(bench_dir)

    if not samples:
        print("ERROR: No benchmark samples found. Run with --generate first.")
        return None

    print(f"\nRunning evaluation: label='{label}', samples={len(samples)}")

    adapter = RecognizerAdapter()
    all_comparisons = []
    per_subset = {}

    for i, sample in enumerate(samples):
        img_path = sample["image_path"]
        gt = sample["gt_intersections"]
        subset = sample["subset"]
        sid = sample["sample_id"]

        print(f"  [{i+1}/{len(samples)}] {subset}/{sid} ... ", end="", flush=True)

        try:
            t0 = time.time()
            coords, result, _ = adapter.recognize_intersections(img_path)
            elapsed = time.time() - t0

            comparison = compare_results(gt, coords)
            metrics = compute_metrics(comparison)

            print(f"GT={comparison['num_gt']} Det={comparison['num_detected']} "
                  f"Correct={comparison['num_correct']} F1={metrics['f1']:.3f} ({elapsed:.1f}s)")

            record = {
                **comparison,
                **metrics,
                "subset": subset,
                "sample_id": sid,
                "elapsed": round(elapsed, 2),
                # 移除 set 类型 (不能 JSON 序列化)
                "gt_set": [list(p) for p in comparison.get("gt_set", set())],
                "detected_set": [list(p) for p in comparison.get("detected_set", set())],
                "correct": [list(p) for p in comparison.get("correct", set())],
                "missed": [list(p) for p in comparison.get("missed", set())],
                "extra": [list(p) for p in comparison.get("extra", set())],
            }
            all_comparisons.append(record)

            if subset not in per_subset:
                per_subset[subset] = []
            per_subset[subset].append(comparison)

        except Exception as e:
            print(f"ERROR: {e}")
            all_comparisons.append({
                "subset": subset, "sample_id": sid,
                "error": str(e),
                "num_gt": len(gt), "num_detected": 0, "num_correct": 0,
                "num_missed": len(gt), "num_extra": 0,
            })

    # 聚合
    overall = aggregate_metrics([c for c in all_comparisons if "error" not in c])
    subset_metrics = {}
    for subset, comps in per_subset.items():
        subset_metrics[subset] = aggregate_metrics(comps)

    report = {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(samples),
        "overall": overall,
        "per_subset": subset_metrics,
        "details": all_comparisons,
    }

    # 保存
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{label}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    # 打印摘要
    _print_summary(report)
    return report


def _print_summary(report):
    """打印评估摘要"""
    label = report["label"]
    o = report["overall"]

    print(f"\n{'='*60}")
    print(f"  {label.upper()} — Overall Metrics ({o['num_images']} images)")
    print(f"{'='*60}")
    print(f"  GT points:       {o['total_gt']}")
    print(f"  Detected points: {o['total_detected']}")
    print(f"  Correct:         {o['total_correct']}")
    print(f"  Missed:          {o['total_missed']}")
    print(f"  Extra (FP):      {o['total_extra']}")
    print(f"")
    print(f"  Micro Jaccard:   {o['micro_jaccard']:.4f}")
    print(f"  Micro Precision: {o['micro_precision']:.4f}")
    print(f"  Micro Recall:    {o['micro_recall']:.4f}")
    print(f"  Micro F1:        {o['micro_f1']:.4f}")
    print(f"")
    print(f"  Mean Jaccard:    {o['mean_jaccard']:.4f}")
    print(f"  Mean F1:         {o['mean_f1']:.4f}")

    print(f"\n  Per-Subset:")
    for subset, m in report.get("per_subset", {}).items():
        print(f"    {subset:30s}  F1={m['mean_f1']:.4f}  "
              f"Prec={m['mean_precision']:.4f}  Rec={m['mean_recall']:.4f}  "
              f"({m['num_images']} imgs)")


def compare_reports(path_a: str, path_b: str):
    """对比两份报告"""
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))

    print(f"\n{'='*70}")
    print(f"  COMPARISON: {a['label']} vs {b['label']}")
    print(f"{'='*70}")

    oa, ob = a["overall"], b["overall"]

    headers = ["Metric", a["label"], b["label"], "Delta"]
    rows = []
    for key in ["micro_jaccard", "micro_precision", "micro_recall", "micro_f1",
                 "mean_jaccard", "mean_f1", "total_correct", "total_missed", "total_extra"]:
        va = oa.get(key, 0)
        vb = ob.get(key, 0)
        if isinstance(va, float):
            delta = vb - va
            mark = "OK" if delta >= -0.01 else "REGRESS"
            rows.append((key, f"{va:.4f}", f"{vb:.4f}", f"{delta:+.4f} {mark}"))
        else:
            delta = vb - va
            mark = "OK" if key != "total_missed" or delta <= 0 else "REGRESS"
            rows.append((key, str(va), str(vb), f"{delta:+d} {mark}"))

    # 打印表格
    col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(f"\n  {fmt.format(*headers)}")
    print(f"  {'-' * sum(col_w + [2*(len(col_w)-1)])}")
    for r in rows:
        print(f"  {fmt.format(*r)}")

    # Per-subset 对比
    all_subsets = set(a.get("per_subset", {}).keys()) | set(b.get("per_subset", {}).keys())
    if all_subsets:
        print(f"\n  Per-Subset F1:")
        for subset in sorted(all_subsets):
            f1_a = a.get("per_subset", {}).get(subset, {}).get("mean_f1", 0)
            f1_b = b.get("per_subset", {}).get(subset, {}).get("mean_f1", 0)
            delta = f1_b - f1_a
            mark = "OK" if delta >= -0.02 else "!! REGRESSION"
            print(f"    {subset:30s}  {f1_a:.4f} → {f1_b:.4f}  ({delta:+.4f}) {mark}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Recognizer M0 regression test")
    parser.add_argument("--generate", action="store_true", help="Generate benchmark first")
    parser.add_argument("--label", default="eval", help="Label for this evaluation run")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two report JSONs")
    parser.add_argument("--bench-dir", default=str(BENCH_DIR), help="Benchmark directory")
    args = parser.parse_args()

    if args.compare:
        compare_reports(args.compare[0], args.compare[1])
        return

    if args.generate:
        generate_benchmark()

    run_evaluation(args.bench_dir, label=args.label)


if __name__ == "__main__":
    main()
