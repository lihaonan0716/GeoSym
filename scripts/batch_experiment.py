"""Batch experiment runner with real-time progress tracking.

Runs N samples per family across all 7 conditions (E0-E5, E5_raw), saving
progress after every single sample so that external monitors can
check how far along we are at any time.

Results are persisted to configurable local output files so monitors can load
them directly without depending on private repository paths.

Usage:
    python scripts/batch_experiment.py --samples-per-family 5
    python scripts/batch_experiment.py --samples-per-family 30 --model GPT-5.5 --judge-model Claude-Opus-4.8
    python scripts/batch_experiment.py --samples-per-family 5 --family linear
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

FAMILIES = [
    "absolute", "cubic", "exponential", "linear",
    "logarithmic", "piecewise_linear", "quadratic", "rational",
]
PROFILE = "clean_render"
# `E5` (M0+M1+M2+M3) retired 2026-09-19 with M1/M2. `E5_raw` (M0+M3) is now
# GeoSym itself, not an ablation -- call it "GeoSym" in prose, but never rename
# the string: `E5` and `E5_raw` coexist in the same paid artifacts, so renaming
# would silently read the retired arm. See CLAUDE.md "Experiment matrix".
#
# Two lists, because one list cannot mean both things:
#   ALL_CONDITIONS      -- what a new run may execute. E5 is gone; asking for it
#                          must fail loudly rather than quietly cost money.
#   HISTORICAL_CONDITIONS -- what the 29425 paid rows on disk contain. Analysis
#                          and rescoring iterate over THIS, or they would
#                          silently drop the 3439 historical E5 rows and report
#                          a subset as if it were the whole artifact.
ALL_CONDITIONS = ["E0", "E1", "E2", "E3", "E4", "E5_raw"]

RETIRED_CONDITIONS = ["E5"]

HISTORICAL_CONDITIONS = ["E0", "E1", "E2", "E3", "E4", "E5", "E5_raw"]

PROGRESS_FILE = Path(
    os.environ.get("GEOSYM_BATCH_PROGRESS_FILE", str(_project_root / "outputs" / "batch_progress.json"))
)
RESULTS_FILE = Path(
    os.environ.get("GEOSYM_BATCH_RESULTS_FILE", str(_project_root / "outputs" / "batch_results.json"))
)


def _select_samples(dataset_dir: Path, family: str, n: int) -> list[str]:
    """Return up to n image_names for a family, sorted alphabetically."""
    subset_dir = dataset_dir / "images" / f"{family}-{PROFILE}"
    if not subset_dir.exists():
        return []
    files = sorted(f for f in subset_dir.iterdir() if f.suffix == ".png")[:n]
    result = []
    for f in files:
        rel = f.relative_to(dataset_dir / "images")
        result.append(str(rel.with_suffix("")).replace("\\", "/"))
    return result


def _parse_score(md: str) -> float | None:
    """Extract Total Score from condition markdown."""
    import re
    if not md:
        return None
    m = re.search(r"\*\*(?:Total Score|总分)\*\*[：:]\s*([\d.]+)", md)
    if m:
        return float(m.group(1))
    m = re.search(r"\*\*([\d.]+)\*\*", md)
    if m:
        return float(m.group(1))
    return None


def _save_progress(progress: dict):
    """Atomically write progress file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    tmp.replace(PROGRESS_FILE)


def _local_manifest_path(run_id: str) -> Path:
    return _project_root / "outputs" / "batch_manifests" / f"{run_id}.json"


def build_run_manifest(
    images: list[str],
    reasoning_model: str,
    judge_model: str,
    archive_outputs: bool,
    text_only_l2l3: bool,
    gemini_enabled: bool,
) -> dict:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "schema_version": "batch_experiment_manifest_v1",
        "run_id": f"batch_{timestamp}",
        "images": images,
        "reasoning_model": reasoning_model,
        "judge_model": judge_model,
        "archive_outputs": archive_outputs,
        "text_only_l2l3": text_only_l2l3,
        "gemini_enabled": gemini_enabled,
        "rows": {},
        "log": [],
    }


def save_run_manifest(manifest: dict) -> None:
    path = _local_manifest_path(str(manifest["run_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_reasoning_record(image_name: str, model_name: str) -> dict | None:
    return None


def summarize_record_row(record: dict) -> dict:
    return record


def archive_reasoning_record(image_name: str, model_name: str, run_id: str) -> None:
    return None


def run_batch(samples_per_family: int, dataset_dir: Path,
              target_family: str | None = None,
              model_override: str | None = None,
              judge_model_override: str | None = None,
              archive: bool = True):
    from app.reasoning import run_reasoning_all_conditions
    from app.helpers import PROVIDER_MAP, _get_saved_key

    # Find reasoning model
    if model_override:
        model_name = model_override
        api_key = _get_saved_key(model_name)
        if not api_key:
            raise RuntimeError(f"No API key for model '{model_name}'")
    else:
        model_name = api_key = None
        for prov, models in PROVIDER_MAP.items():
            for m in models:
                k = _get_saved_key(m)
                if k:
                    model_name, api_key = m, k
                    break
            if model_name:
                break
        if not model_name:
            raise RuntimeError("No model with API key found")

    # Find judge model (defaults to reasoning model if not specified)
    if judge_model_override:
        judge_model_name = judge_model_override
        judge_api_key = _get_saved_key(judge_model_name)
        if not judge_api_key:
            raise RuntimeError(f"No API key for judge model '{judge_model_name}'")
    else:
        judge_model_name = model_name
        judge_api_key = api_key

    families = [target_family] if target_family else FAMILIES
    total_families = len(families)

    # Build sample list
    all_samples: list[tuple[str, str]] = []  # (family, image_name)
    for fam in families:
        names = _select_samples(dataset_dir, fam, samples_per_family)
        for name in names:
            all_samples.append((fam, name))

    all_image_names = [name for _, name in all_samples]
    total = len(all_samples)
    print(f"Reasoning model: {model_name}")
    print(f"Judge model: {judge_model_name}")
    print(f"Families: {families}")
    print(f"Total samples: {total} ({samples_per_family}/family × {total_families} families)")
    run_conditions = ALL_CONDITIONS
    print(f"Conditions: {run_conditions}")
    print(f"Archive: {archive}")
    print()

    # ---- Create batch manifest ----
    manifest = build_run_manifest(
        images=all_image_names,
        reasoning_model=model_name,
        judge_model=judge_model_name,
        archive_outputs=archive,
        text_only_l2l3=False,
        gemini_enabled=False,
    )
    run_id = manifest["run_id"]
    manifest["started_at"] = datetime.now(timezone.utc).isoformat()
    manifest["run_status"] = "running"
    manifest.setdefault("log", []).append(
        f"[{manifest['started_at']}] CLI batch started: {total} images, "
        f"reasoning={model_name}, judge={judge_model_name}"
    )
    save_run_manifest(manifest)
    print(f"Manifest: {run_id}")
    print()

    # ---- Initialize lightweight progress (for real-time monitoring) ----
    progress = {
        "status": "running",
        "run_id": run_id,
        "reasoning_model": model_name,
        "judge_model": judge_model_name,
        "total_samples": total,
        "completed_samples": 0,
        "current_family": "",
        "current_sample": "",
        "started_at": manifest["started_at"],
        "last_update": "",
        "eta_minutes": None,
        "family_progress": {f: {"total": 0, "done": 0} for f in families},
        "results": [],
    }
    for fam in families:
        progress["family_progress"][fam]["total"] = len(
            [s for s in all_samples if s[0] == fam]
        )
    _save_progress(progress)

    start_time = time.time()

    for idx, (family, image_name) in enumerate(all_samples):
        progress["current_family"] = family
        progress["current_sample"] = image_name
        progress["last_update"] = datetime.now(timezone.utc).isoformat()
        _save_progress(progress)

        # Update manifest state to running
        state = manifest["image_status"].setdefault(image_name, {
            "image_name": image_name, "status": "pending",
            "attempts": 0, "last_error": "", "updated_at": "",
            "row_summary": None,
        })
        state["status"] = "running"
        state["updated_at"] = datetime.now(timezone.utc).isoformat()

        sample_start = time.time()
        print(f"[{idx+1}/{total}] {image_name} ...", flush=True)

        try:
            gen = run_reasoning_all_conditions(
                image_name=image_name,
                model_name=model_name,
                api_key=api_key,
                judge_model_name=judge_model_name,
                judge_api_key=judge_api_key,
                lang="en",
                force_rerun=True,
                target_conditions=run_conditions,
            )

            last_yield = None
            for result in gen:
                last_yield = result

            scores = {}
            if last_yield and len(last_yield) >= 9:
                # Yield tuple: (gt_md, progress, E0, E1, E2, E3, E4, E5, E5_raw)
                cond_map = {
                    "E0": 2, "E1": 3, "E2": 4, "E3": 5, "E4": 6, "E5": 7,
                    "E5_raw": 8,
                }
                for cond, idx_pos in cond_map.items():
                    md = last_yield[idx_pos] if idx_pos < len(last_yield) else ""
                    scores[cond] = _parse_score(md)

            # Update manifest state to success
            record = load_reasoning_record(image_name, model_name)
            state["status"] = "success"
            state["last_error"] = ""
            state["attempts"] = 1
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            if record:
                state["row_summary"] = summarize_record_row(record)

            # Archive
            if archive:
                archive_reasoning_record(image_name, model_name, run_id)

        except Exception as exc:
            scores = {}
            state["status"] = "error"
            state["last_error"] = str(exc)
            state["attempts"] = 1
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            print(f"         ERROR: {exc}", flush=True)

        elapsed_sample = time.time() - sample_start
        score_str = " | ".join(
            f"{c}={v:.3f}" if v is not None else f"{c}=—"
            for c, v in scores.items()
        )
        if scores:
            print(f"         {score_str}  ({elapsed_sample:.0f}s)", flush=True)

        # Update progress
        progress["completed_samples"] = idx + 1
        progress["family_progress"][family]["done"] += 1
        elapsed_total = time.time() - start_time
        remaining = total - (idx + 1)
        avg_per_sample = elapsed_total / (idx + 1)
        progress["eta_minutes"] = round(remaining * avg_per_sample / 60, 1)
        progress["results"].append({
            "family": family,
            "image_name": image_name,
            "scores": scores,
            "duration_s": round(elapsed_sample, 1),
        })
        progress["last_update"] = datetime.now(timezone.utc).isoformat()
        _save_progress(progress)

        # Persist manifest after each image
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest.setdefault("log", []).append(
            f"[{manifest['updated_at']}] [{idx+1}/{total}] {image_name} → {state['status']}"
        )
        save_run_manifest(manifest)

    # ---- Finalize ----
    progress["status"] = "completed"
    progress["eta_minutes"] = 0
    progress["current_sample"] = ""
    progress["current_family"] = ""
    progress["last_update"] = datetime.now(timezone.utc).isoformat()
    total_elapsed = time.time() - start_time
    progress["total_duration_minutes"] = round(total_elapsed / 60, 1)
    _save_progress(progress)

    manifest["run_status"] = "completed"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["duration_ms"] = round(total_elapsed * 1000, 3)
    manifest.setdefault("log", []).append(
        f"[{manifest['finished_at']}] Batch completed: {total} images in {total_elapsed/60:.1f} min"
    )
    save_run_manifest(manifest)

    # Also save full results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {total} samples in {total_elapsed/60:.1f} min")
    print(f"Manifest: {run_id}")
    print(f"Progress: {PROGRESS_FILE}")
    print(f"Results: {RESULTS_FILE}")
    return progress


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--samples-per-family", type=int, default=5)
    parser.add_argument("--family", type=str, default=None,
                        help="Run only one family (e.g., 'linear')")
    parser.add_argument("--model", type=str, default=None,
                        help="Reasoning model name (e.g., 'GPT-5.5'). Default: auto-detect.")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Judge model name (e.g., 'Claude-Opus-4.8'). Default: same as --model.")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--no-archive", action="store_true",
                        help="Skip archiving condition records.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = _project_root / dataset_dir

    run_batch(args.samples_per_family, dataset_dir, args.family, args.model,
              args.judge_model, archive=not args.no_archive)


if __name__ == "__main__":
    main()
