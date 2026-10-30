"""
Presenter experiment recording and aggregation.

Collects Direct-LLM vs GeoSym results from reports/presenter/,
aggregates pairwise comparison outcomes, and produces a consolidated
summary for paper figures. No automated scoring — the cross-domain
evaluation is qualitative (case study).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parent.parent
_ASSETS_ROOT = _root / "assets" / "real_charts"
_REPORTS_ROOT = _root / "reports" / "presenter"


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def _find_report_dir(sample_id: str, model_pattern: str = "GPT-5.4") -> Optional[Path]:
    """Find the best matching report directory for a sample."""
    sample_dir = _REPORTS_ROOT / sample_id
    if not sample_dir.exists():
        return None
    preferred = sample_dir / f"{model_pattern}__cmp__GPT-5.4"
    if preferred.exists():
        return preferred
    for d in sorted(sample_dir.iterdir()):
        if d.is_dir():
            return d
    return None


def _load_meta(sample_id: str) -> Optional[Dict[str, Any]]:
    """Load meta.json for a sample."""
    family = sample_id.rsplit("_", 1)[0]
    meta_path = _ASSETS_ROOT / family / sample_id / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def collect_single(sample_id: str, model_pattern: str = "GPT-5.4") -> Optional[Dict[str, Any]]:
    """Collect results for a single sample (no scoring)."""
    meta = _load_meta(sample_id)
    if meta is None:
        logger.warning(f"Meta not found: {sample_id}")
        return None

    report_dir = _find_report_dir(sample_id, model_pattern)
    if report_dir is None:
        return {
            "sample_id": sample_id,
            "family": meta.get("source_family", ""),
            "title": meta.get("title", ""),
            "question": meta.get("question", ""),
            "gold_answer": meta.get("gold_answer", {}),
            "status": "missing",
        }

    # Load results
    direct_path = report_dir / "direct_result.json"
    geosym_path = report_dir / "geosym_final.json"
    pairwise_path = report_dir / "pairwise_compare.json"

    direct_data = json.loads(direct_path.read_text(encoding="utf-8")) if direct_path.exists() else None
    geosym_data = json.loads(geosym_path.read_text(encoding="utf-8")) if geosym_path.exists() else None
    pairwise_data = json.loads(pairwise_path.read_text(encoding="utf-8")) if pairwise_path.exists() else None

    # Extract answers
    direct_answer = ""
    direct_evidence = []
    if direct_data:
        pr = direct_data.get("parsed_response", direct_data.get("parsed", {}))
        direct_answer = pr.get("answer", "")
        direct_evidence = pr.get("evidence", [])

    geosym_answer = ""
    geosym_evidence = []
    if geosym_data:
        pr = geosym_data.get("parsed", geosym_data.get("parsed_response", {}))
        geosym_answer = pr.get("answer", "")
        geosym_evidence = pr.get("evidence", [])

    # Pairwise
    pairwise_winner = ""
    pairwise_summary = ""
    pairwise_dimensions = []
    pairwise_justification = []
    if pairwise_data:
        pp = pairwise_data.get("parsed", pairwise_data)
        pairwise_winner = pp.get("winner", "")
        pairwise_summary = pp.get("summary", "")
        pairwise_dimensions = pp.get("improvement_dimensions", [])
        pairwise_justification = pp.get("justification", [])

    return {
        "sample_id": sample_id,
        "family": meta.get("source_family", ""),
        "title": meta.get("title", ""),
        "question": meta.get("question", ""),
        "gold_answer": meta.get("gold_answer", {}),
        "model_config": report_dir.name,
        "status": "complete",
        "direct": {
            "answer": direct_answer,
            "evidence": direct_evidence,
        },
        "geosym": {
            "answer": geosym_answer,
            "evidence": geosym_evidence,
        },
        "pairwise": {
            "winner": pairwise_winner,
            "summary": pairwise_summary,
            "improvement_dimensions": pairwise_dimensions,
            "justification": pairwise_justification,
        },
    }


def collect_all(model_pattern: str = "GPT-5.4") -> Dict[str, Any]:
    """Collect results for all samples and produce an aggregated summary."""
    all_samples = []
    for family in ["omniscience"]:
        family_dir = _ASSETS_ROOT / family
        if not family_dir.exists():
            continue
        for sample_dir in sorted(family_dir.iterdir()):
            if sample_dir.is_dir() and (sample_dir / "meta.json").exists():
                result = collect_single(sample_dir.name, model_pattern)
                if result:
                    all_samples.append(result)

    complete = [s for s in all_samples if s["status"] == "complete"]
    missing = [s for s in all_samples if s["status"] == "missing"]

    # Pairwise stats per family
    by_family = {}
    for s in complete:
        fam = s["family"]
        if fam not in by_family:
            by_family[fam] = {"pairwise_wins": 0, "count": 0}
        winner = s["pairwise"]["winner"].upper()
        if winner in ("GEOSYM", "GEOSYM_ASSISTED"):
            by_family[fam]["pairwise_wins"] += 1
        by_family[fam]["count"] += 1

    family_stats = {}
    for fam, data in by_family.items():
        family_stats[fam] = {
            "count": data["count"],
            "pairwise_geosym_wins": data["pairwise_wins"],
        }

    n_total = len(complete)
    total_wins = sum(
        1 for s in complete
        if s["pairwise"]["winner"].upper() in ("GEOSYM", "GEOSYM_ASSISTED")
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(all_samples),
        "complete": len(complete),
        "missing": len(missing),
        "missing_ids": [s["sample_id"] for s in missing],
        "overall": {
            "pairwise_geosym_wins": total_wins,
            "pairwise_total": n_total,
        },
        "by_family": family_stats,
        "samples": all_samples,
    }
    return summary


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_summary(output_path: Optional[str] = None) -> Path:
    """Collect all results and save to a JSON summary file."""
    summary = collect_all()
    if output_path is None:
        out = _REPORTS_ROOT / "experiment_summary.json"
    else:
        out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Summary saved to {out} ({summary['complete']}/{summary['total_samples']} complete)")
    return out


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_missing(
    analysis_model: str = "GPT-5.4",
    comparison_model: str = "GPT-5.4",
) -> List[str]:
    """Run presenter showcase for samples that have no reports yet.

    Returns list of sample_ids that were run.
    """
    from .service import list_real_chart_samples, run_presenter_showcase

    all_ids = list_real_chart_samples()
    ran = []
    for sid in all_ids:
        report_dir = _find_report_dir(sid)
        if report_dir is not None:
            continue
        logger.info(f"Running missing sample: {sid}")
        try:
            run_presenter_showcase(sid, analysis_model, comparison_model)
            ran.append(sid)
            logger.info(f"  -> {sid} complete")
        except Exception as exc:
            logger.error(f"  -> {sid} failed: {exc}")
    return ran


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Presenter experiment tools")
    sub = parser.add_subparsers(dest="command")

    p_collect = sub.add_parser("collect", help="Collect all results into summary JSON")
    p_collect.add_argument("-o", "--output", default=None, help="Output JSON path")

    p_run = sub.add_parser("run-missing", help="Run showcase for missing samples")
    p_run.add_argument("--model", default="GPT-5.4", help="Analysis model")
    p_run.add_argument("--judge", default="GPT-5.4", help="Comparison model")

    p_show = sub.add_parser("show", help="Show summary of collected results")

    args = parser.parse_args()

    if args.command == "collect":
        path = save_summary(args.output)
        print(f"Saved to {path}")

    elif args.command == "run-missing":
        ran = run_missing(args.model, args.judge)
        if ran:
            print(f"Ran {len(ran)} samples: {ran}")
            path = save_summary()
            print(f"Summary updated: {path}")
        else:
            print("No missing samples found.")

    elif args.command == "show":
        summary = collect_all()
        print(f"Total: {summary['complete']}/{summary['total_samples']} complete")
        if summary["missing_ids"]:
            print(f"Missing: {summary['missing_ids']}")
        wins = summary["overall"]["pairwise_geosym_wins"]
        total = summary["overall"]["pairwise_total"]
        print(f"\nPairwise: GeoSym wins {wins}/{total}")
        for fam, stats in summary["by_family"].items():
            print(f"  {fam}: GeoSym wins {stats['pairwise_geosym_wins']}/{stats['count']}")

        # Show per-sample detail
        print(f"\n{'ID':<20} {'Family':<14} {'Winner':<20} {'Dimensions'}")
        print("-" * 85)
        for s in summary["samples"]:
            if s["status"] != "complete":
                print(f"{s['sample_id']:<20} {s['family']:<14} {'(missing)':<20}")
                continue
            dims = ", ".join(s["pairwise"]["improvement_dimensions"][:3])
            print(f"{s['sample_id']:<20} {s['family']:<14} "
                  f"{s['pairwise']['winner']:<20} {dims}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
