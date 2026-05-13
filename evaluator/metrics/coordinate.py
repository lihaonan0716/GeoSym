"""
Metrics for comparing detected intersections to ground truth.

Supports two matching modes:
  - Exact match: integer coordinate equality (default).
  - Tolerance match: Euclidean distance within a configurable epsilon,
    using greedy nearest-neighbour assignment.

Metrics:
  - Jaccard (IoU) = |GT ∩ Det| / |GT ∪ Det|  (set similarity)
  - Precision     = correct / detected        (how many detections are right)
  - Recall        = correct / gt              (how many GT points were found)
  - F1            = 2*P*R / (P+R)             (harmonic mean)
  - FPR           = extra / detected          (false positive rate, for negative samples)
"""

import math
from typing import Dict, List, Optional, Set, Tuple


def _to_set(coords: List[Tuple[int, int]] | List[dict]) -> Set[Tuple[int, int]]:
    """Convert coordinate list to set of (x, y) tuples."""
    out = set()
    for c in coords:
        if isinstance(c, dict):
            x = int(c.get("coordinate_x", 0))
            y = int(c.get("coordinate_y", 0))
        else:
            x, y = int(c[0]), int(c[1])
        out.add((x, y))
    return out


def _to_float_list(coords) -> List[Tuple[float, float]]:
    """Convert coordinates to list of (x, y) float tuples."""
    out = []
    for c in coords:
        if isinstance(c, dict):
            out.append((float(c.get("coordinate_x", 0)), float(c.get("coordinate_y", 0))))
        else:
            out.append((float(c[0]), float(c[1])))
    return out


def compare_results(
    gt_coords: List[dict] | List[Tuple[int, int]],
    detected_coords: List[Tuple[int, int]],
) -> Dict[str, any]:
    """
    Compare detected coordinates to ground truth.
    All coordinates are integers; comparison is exact match.
    """
    gt_set = _to_set(gt_coords)
    det_set = _to_set(detected_coords) if detected_coords else set()

    correct = gt_set & det_set
    missed = gt_set - det_set
    extra = det_set - gt_set

    return {
        "gt_set": gt_set,
        "detected_set": det_set,
        "correct": correct,
        "missed": missed,
        "extra": extra,
        "num_gt": len(gt_set),
        "num_detected": len(det_set),
        "num_correct": len(correct),
        "num_missed": len(missed),
        "num_extra": len(extra),
    }


def compare_results_tolerance(
    gt_coords,
    detected_coords,
    epsilon: float = 0.5,
) -> Dict[str, any]:
    """
    Tolerance-based matching: a detection matches a GT point if their
    Euclidean distance <= epsilon. Uses greedy nearest-neighbour assignment
    (each GT and detection can match at most once).

    Args:
        gt_coords: Ground truth coordinate list.
        detected_coords: Detected coordinate list.
        epsilon: Maximum Euclidean distance for a match.

    Returns:
        Same schema as compare_results, plus 'match_distances'.
    """
    gt_pts = _to_float_list(gt_coords) if gt_coords else []
    det_pts = _to_float_list(detected_coords) if detected_coords else []

    matched_gt: Set[int] = set()
    matched_det: Set[int] = set()
    match_distances: List[float] = []

    pairs = []
    for i, (gx, gy) in enumerate(gt_pts):
        for j, (dx, dy) in enumerate(det_pts):
            d = math.hypot(gx - dx, gy - dy)
            if d <= epsilon:
                pairs.append((d, i, j))
    pairs.sort()

    for d, i, j in pairs:
        if i not in matched_gt and j not in matched_det:
            matched_gt.add(i)
            matched_det.add(j)
            match_distances.append(d)

    num_correct = len(matched_gt)
    return {
        "num_gt": len(gt_pts),
        "num_detected": len(det_pts),
        "num_correct": num_correct,
        "num_missed": len(gt_pts) - num_correct,
        "num_extra": len(det_pts) - num_correct,
        "match_distances": match_distances,
        "epsilon": epsilon,
    }


def compute_metrics(comparison: Dict) -> Dict[str, float]:
    """
    Compute Jaccard (IoU), precision, recall, and F1 from comparison result.

    - Jaccard (IoU) = |GT ∩ Det| / |GT ∪ Det| = correct / (gt + det - correct)
    - Precision     = correct / detected
    - Recall        = correct / gt
    - F1            = 2*P*R / (P+R)

    Returns:
        {
            "jaccard": float,
            "precision": float,
            "recall": float,
            "f1": float,
        }
    """
    num_gt = comparison["num_gt"]
    num_detected = comparison["num_detected"]
    num_correct = comparison["num_correct"]

    # |GT ∪ Det| = num_gt + num_detected - num_correct
    union = num_gt + num_detected - num_correct
    jaccard = num_correct / union if union > 0 else 0.0
    precision = num_correct / num_detected if num_detected > 0 else 0.0
    recall = num_correct / num_gt if num_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Aggregate metrics from multiple images.

    Args:
        results: List of comparison dicts (from compare_results)

    Returns:
        Aggregated metrics: macro & micro Jaccard/Precision/Recall/F1, plus total counts.
    """
    if not results:
        return {
            "num_images": 0,
            "total_gt": 0,
            "total_detected": 0,
            "total_correct": 0,
            "total_missed": 0,
            "total_extra": 0,
            "mean_jaccard": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f1": 0.0,
            "micro_jaccard": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
        }

    total_gt = sum(r["num_gt"] for r in results)
    total_detected = sum(r["num_detected"] for r in results)
    total_correct = sum(r["num_correct"] for r in results)
    total_missed = sum(r.get("num_missed", r["num_gt"] - r["num_correct"]) for r in results)
    total_extra = sum(r["num_extra"] for r in results)

    # Per-image metrics then average (macro)
    jaccards = []
    precisions = []
    recalls = []
    f1s = []
    for r in results:
        m = compute_metrics(r)
        jaccards.append(m["jaccard"])
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])

    n = len(results)

    # Micro metrics (from totals)
    total_union = total_gt + total_detected - total_correct
    micro_jaccard = total_correct / total_union if total_union > 0 else 0.0
    micro_precision = total_correct / total_detected if total_detected > 0 else 0.0
    micro_recall = total_correct / total_gt if total_gt > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) \
        if (micro_precision + micro_recall) > 0 else 0.0

    return {
        "num_images": n,
        "total_gt": total_gt,
        "total_detected": total_detected,
        "total_correct": total_correct,
        "total_missed": total_missed,
        "total_extra": total_extra,
        "mean_jaccard": sum(jaccards) / n,
        "mean_precision": sum(precisions) / n,
        "mean_recall": sum(recalls) / n,
        "mean_f1": sum(f1s) / n,
        "micro_jaccard": micro_jaccard,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }
