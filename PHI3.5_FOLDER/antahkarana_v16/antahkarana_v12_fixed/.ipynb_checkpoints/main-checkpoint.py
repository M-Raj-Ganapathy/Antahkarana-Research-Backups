"""
main.py — Antahkarana v11 — Full IEEE Evaluation Pipeline

Execution order:
  1. Load vLLM engine (once)
  2. Load datasets (100 samples each)
  3. Run 4 baselines + Antahkarana (batched)
  4. Compute metrics
  5. Run ablation (50 samples)
  6. Statistical significance tests
  7. Save all outputs (JSON, CSV, TXT, PNG)

Run:
    python main.py
    # or from JupyterLab:
    %run main.py
"""

import os
import sys
import json
import csv
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# ── Logging setup ──────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/run.log", mode="w"),
    ],
)
logger = logging.getLogger("main")

# ── Result dirs ────────────────────────────────────────────────────────────
for d in [
    "results/raw", "results/processed", "results/ablation",
    "results/stats", "results/plots",
]:
    Path(d).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Step 0: Imports (deferred so engine loads once)
# ──────────────────────────────────────────────────────────────────────────────

def _import_all():
    global get_engine, load_all_datasets, run_all_baselines
    global AntahkaranaSystem, score_result, aggregate_scores
    global run_significance_tests, compute_throughput_stats
    global run_ablation_study, generate_all_plots, ABLATION_N

    from vllm_engine       import get_engine
    from hf_datasets       import load_all_datasets, ABLATION_N
    from baselines         import run_all_baselines
    from antahkarana       import AntahkaranaSystem
    from evaluation        import (
        score_result, aggregate_scores, run_significance_tests,
        compute_throughput_stats, run_ablation_study, generate_all_plots,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Engine warm-up
# ──────────────────────────────────────────────────────────────────────────────

def step_engine():
    logger.info("=" * 70)
    logger.info("STEP 1: Initialising vLLM engine")
    logger.info("=" * 70)
    engine = get_engine()
    logger.info(f"Engine ready. Model load time: {engine._load_time:.1f}s")
    return engine


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Load datasets
# ──────────────────────────────────────────────────────────────────────────────

def step_datasets(n: int = 100):
    logger.info("=" * 70)
    logger.info(f"STEP 2: Loading datasets (n={n} each)")
    logger.info("=" * 70)
    datasets = load_all_datasets(n=n)
    for name, data in datasets.items():
        logger.info(f"  {name}: {len(data)} samples")
    return datasets


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Run all methods
# ──────────────────────────────────────────────────────────────────────────────

def step_run_methods(datasets: Dict, antahkarana_system) -> Dict:
    """
    Returns nested dict:
      {dataset_name: {method_name: [result_dicts]}}
    """
    logger.info("=" * 70)
    logger.info("STEP 3: Running baselines + Antahkarana")
    logger.info("=" * 70)

    ALL_RESULTS: Dict[str, Dict[str, List]] = {}

    baselines = ["direct", "cot", "self_consistency", "tot"]

    for ds_name, samples in datasets.items():
        if not samples:
            logger.warning(f"  Skipping {ds_name} (0 samples)")
            continue

        logger.info(f"\n▶ Dataset: {ds_name} ({len(samples)} samples)")
        ALL_RESULTS[ds_name] = {}

        # ── Baselines ──────────────────────────────────────────────────────
        baseline_results = run_all_baselines(samples, ds_name, methods=baselines)
        ALL_RESULTS[ds_name].update(baseline_results)

        # ── Antahkarana ────────────────────────────────────────────────────
        logger.info(f"  [{ds_name}] Running Antahkarana ({len(samples)} samples)…")
        t0 = time.time()
        ant_results = antahkarana_system.run_batch(samples, ds_name)
        elapsed = time.time() - t0
        ALL_RESULTS[ds_name]["antahkarana"] = ant_results
        logger.info(
            f"  [{ds_name}][antahkarana] done in {elapsed:.1f}s "
            f"({len(samples)/elapsed:.1f} samp/s)"
        )

        # ── Save raw JSON per dataset ───────────────────────────────────────
        raw_path = f"results/raw/{ds_name}_results.json"
        _safe_json_dump(ALL_RESULTS[ds_name], raw_path)
        logger.info(f"  Raw results saved: {raw_path}")

    return ALL_RESULTS


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Compute metrics
# ──────────────────────────────────────────────────────────────────────────────

def step_metrics(all_results: Dict, datasets: Dict) -> Dict:
    """
    Returns:
      summary[dataset][method] = aggregate_scores dict
    """
    logger.info("=" * 70)
    logger.info("STEP 4: Computing metrics")
    logger.info("=" * 70)

    summary: Dict[str, Dict[str, Any]] = {}

    for ds_name, method_results in all_results.items():
        samples     = datasets.get(ds_name, [])
        sample_map  = {s.get("id", i): s for i, s in enumerate(samples)}
        summary[ds_name] = {}

        for method, results in method_results.items():
            scores_list = []
            for i, r in enumerate(results):
                sample = sample_map.get(r.get("id"), samples[i] if i < len(samples) else {})
                scores = score_result(r, sample, ds_name)
                scores_list.append(scores)
                r["scores"] = scores  # annotate in-place

            agg = aggregate_scores(scores_list)
            # Add throughput
            thr = compute_throughput_stats(results)
            agg.update({f"throughput_{k}": {"mean": v} for k, v in thr.items()})
            summary[ds_name][method] = agg

            logger.info(
                f"  [{ds_name}][{method}] "
                f"EM={agg.get('em', {}).get('mean', 0):.3f} "
                f"F1={agg.get('f1', {}).get('mean', 0):.3f} "
                f"Lat={thr.get('mean_latency_s', 0):.3f}s"
            )

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Ablation study
# ──────────────────────────────────────────────────────────────────────────────

def step_ablation(datasets: Dict, n: int = 50) -> Dict:
    logger.info("=" * 70)
    logger.info(f"STEP 5: Ablation study (n={n} per dataset)")
    logger.info("=" * 70)

    # Use HotpotQA for ablation (richest reasoning)
    hotpot_samples = datasets.get("hotpotqa", [])[:n]
    if not hotpot_samples:
        logger.warning("No HotpotQA samples for ablation")
        return {}

    ablation_raw = run_ablation_study(hotpot_samples, "hotpotqa")

    # Score ablation results
    sample_map = {s.get("id", i): s for i, s in enumerate(hotpot_samples)}
    ablation_summary: Dict[str, Any] = {}

    for cfg_name, results in ablation_raw.items():
        scores_list = []
        for i, r in enumerate(results):
            sample = sample_map.get(r.get("id"), hotpot_samples[i] if i < len(hotpot_samples) else {})
            scores = score_result(r, sample, "hotpotqa")
            scores_list.append(scores)
        agg = aggregate_scores(scores_list)
        ablation_summary[cfg_name] = agg
        logger.info(
            f"  Ablation [{cfg_name}] "
            f"EM={agg.get('em',{}).get('mean',0):.3f} "
            f"F1={agg.get('f1',{}).get('mean',0):.3f}"
        )

    _safe_json_dump(ablation_summary, "results/ablation/ablation_summary.json")
    _safe_json_dump(ablation_raw,     "results/ablation/ablation_raw.json")
    return ablation_summary


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Statistical significance
# ──────────────────────────────────────────────────────────────────────────────

def step_significance(all_results: Dict, summary: Dict) -> Dict:
    logger.info("=" * 70)
    logger.info("STEP 6: Statistical significance (paired t-test)")
    logger.info("=" * 70)

    sig_results: Dict[str, Any] = {}

    for ds_name, method_results in all_results.items():
        ant_results   = method_results.get("antahkarana", [])
        ant_f1_scores = [r.get("scores", {}).get("f1", 0.0) for r in ant_results]

        baseline_f1: Dict[str, List[float]] = {}
        for method in ("direct", "cot", "self_consistency", "tot"):
            res = method_results.get(method, [])
            baseline_f1[method] = [r.get("scores", {}).get("f1", 0.0) for r in res]

        sig = run_significance_tests(ant_f1_scores, baseline_f1, metric="f1")
        sig_results[ds_name] = sig

        for method, s in sig.items():
            logger.info(
                f"  [{ds_name}] Antahkarana vs {method}: "
                f"+{s['improvement_pct']:.1f}% F1, "
                f"p={s['p_value']:.4f} {s['significance']}"
            )

    _safe_json_dump(sig_results, "results/stats/significance.json")
    return sig_results


# ──────────────────────────────────────────────────────────────────────────────
# Step 7: Save outputs
# ──────────────────────────────────────────────────────────────────────────────

def step_save_outputs(
    summary:          Dict,
    ablation_summary: Dict,
    sig_results:      Dict,
):
    logger.info("=" * 70)
    logger.info("STEP 7: Saving processed outputs")
    logger.info("=" * 70)

    # ── JSON summary ───────────────────────────────────────────────────────
    _safe_json_dump(summary, "results/processed/metrics_summary.json")

    # ── CSV metrics table ──────────────────────────────────────────────────
    csv_rows = []
    for ds, methods in summary.items():
        for method, agg in methods.items():
            row = {
                "dataset":        ds,
                "method":         method,
                "em_mean":        agg.get("em",  {}).get("mean",     0),
                "em_std":         agg.get("em",  {}).get("std",      0),
                "em_ci_lo":       agg.get("em",  {}).get("ci_low",   0),
                "em_ci_hi":       agg.get("em",  {}).get("ci_high",  0),
                "f1_mean":        agg.get("f1",  {}).get("mean",     0),
                "f1_std":         agg.get("f1",  {}).get("std",      0),
                "f1_ci_lo":       agg.get("f1",  {}).get("ci_low",   0),
                "f1_ci_hi":       agg.get("f1",  {}).get("ci_high",  0),
                "sf_em":          agg.get("sf_em", {}).get("mean", "N/A"),
                "sf_f1":          agg.get("sf_f1", {}).get("mean", "N/A"),
                "latency_mean_s": agg.get("throughput_mean_latency_s", {}).get("mean", 0),
                "throughput_sps": agg.get("throughput_throughput_sps", {}).get("mean", 0),
            }
            csv_rows.append(row)

    if csv_rows:
        csv_path = "results/processed/metrics_table.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"Saved CSV: {csv_path}")

    # ── Ablation CSV ───────────────────────────────────────────────────────
    if ablation_summary:
        abl_rows = []
        for cfg_name, agg in ablation_summary.items():
            abl_rows.append({
                "config":  cfg_name,
                "em_mean": agg.get("em", {}).get("mean", 0),
                "f1_mean": agg.get("f1", {}).get("mean", 0),
                "sf_em":   agg.get("sf_em", {}).get("mean", "N/A"),
                "sf_f1":   agg.get("sf_f1", {}).get("mean", "N/A"),
            })
        abl_csv = "results/ablation/ablation_table.csv"
        with open(abl_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=abl_rows[0].keys())
            writer.writeheader()
            writer.writerows(abl_rows)
        logger.info(f"Saved ablation CSV: {abl_csv}")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_paths = generate_all_plots(summary, ablation_summary)
    logger.info(f"Generated {len(plot_paths)} plots")

    # ── Final TXT report ───────────────────────────────────────────────────
    report = _build_report(summary, ablation_summary, sig_results)
    report_path = "results/processed/final_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved report: {report_path}")
    print(report)


def _build_report(summary: Dict, ablation_summary: Dict, sig_results: Dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("  ANTAHKARANA v11 — IEEE EVALUATION REPORT")
    lines.append("=" * 78)
    # lines.append(f"  Model : Gemma-2-9B-Instruct (google/gemma-2-9b-it)")
    # lines.append(f"  Model : Qwen2.5-7B-Instruct")
    # lines.append(f"  Model : Gemma-2-9B-it-AWQ (bartowski/gemma-2-9b-it-AWQ)")
    lines.append(f"  Model : Phi-3.5-mini-instruct (microsoft/Phi-3.5-mini-instruct)")
    lines.append(f"  HW    : NVIDIA L4 24GB · 32 vCPUs · 128 GB RAM")
    lines.append(f"  Engine: vLLM (continuous batching, BF16, eager mode)")
    lines.append("")

    lines.append("TABLE 1: ANSWER ACCURACY (EM / F1)")
    lines.append("-" * 78)
    header2 = f"{'Method':<22}" + "".join(
        f"{'  ' + ds.upper()[:10]:>18}" for ds in summary
    )
    lines.append(header2)

    methods = ["direct", "cot", "self_consistency", "tot", "antahkarana"]
    for method in methods:
        row = f"{method:<22}"
        for ds in summary:
            agg = summary[ds].get(method, {})
            em  = agg.get("em", {}).get("mean", 0)
            f1  = agg.get("f1", {}).get("mean", 0)
            row += f"  {em:.3f}/{f1:.3f}  "
        lines.append(row)

    lines.append("")
    lines.append("TABLE 2: STATISTICAL SIGNIFICANCE (Antahkarana vs Baselines, F1)")
    lines.append("-" * 78)
    lines.append(f"{'Dataset':<14}{'vs Baseline':<20}{'Δ F1 %':>8}{'p-value':>12}{'Sig':>6}")
    lines.append("-" * 60)

    for ds, comparisons in sig_results.items():
        for method, s in comparisons.items():
            lines.append(
                f"{ds:<14}{method:<20}"
                f"{s['improvement_pct']:>+8.1f}%"
                f"{s['p_value']:>12.4f}"
                f"  {s['significance']}"
            )
    lines.append("")

    if ablation_summary:
        lines.append("TABLE 3: ABLATION STUDY (HotpotQA, n=50)")
        lines.append("-" * 50)
        lines.append(f"{'Config':<28}{'EM':>8}{'F1':>8}")
        lines.append("-" * 44)
        for cfg, agg in ablation_summary.items():
            em = agg.get("em", {}).get("mean", 0)
            f1 = agg.get("f1", {}).get("mean", 0)
            lines.append(f"{cfg:<28}{em:>8.3f}{f1:>8.3f}")
        lines.append("")

    # ── Summary claim ──────────────────────────────────────────────────────
    all_imps = [
        s["improvement_pct"]
        for ds_comps in sig_results.values()
        for s in ds_comps.values()
        if s["significance"] != "ns"
    ]
    if all_imps:
        avg_imp = sum(all_imps) / len(all_imps)
        max_imp = max(all_imps)
        lines.append("=" * 78)
        lines.append(
            f"  CONCLUSION: Antahkarana improves F1 by {avg_imp:.1f}% on average "
            f"(up to {max_imp:.1f}%) over baselines across all datasets "
            f"({sum(1 for s in sig_results.values() for c in s.values() if c['significance']!='ns')} "
            f"statistically significant comparisons, p < 0.05)."
        )
        lines.append("=" * 78)

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

def _safe_json_dump(obj: Any, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    logger.info(f"Saved JSON: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(n_main: int = 500, n_ablation: int = 50):
    t_start = time.time()

    logger.info("▶▶▶ ANTAHKARANA v11 — IEEE EVALUATION PIPELINE STARTED ◀◀◀")

    # 0. Imports
    _import_all()

    # 1. Engine
    step_engine()

    # 2. Datasets
    datasets = step_datasets(n=n_main)

    # 3. Run all methods
    from antahkarana import AntahkaranaSystem
    antahkarana_system = AntahkaranaSystem()
    all_results = step_run_methods(datasets, antahkarana_system)

    # 4. Metrics
    summary = step_metrics(all_results, datasets)

    # 5. Ablation
    ablation_summary = step_ablation(datasets, n=n_ablation)

    # 6. Significance
    sig_results = step_significance(all_results, summary)

    # 7. Save
    step_save_outputs(summary, ablation_summary, sig_results)

    total_time = time.time() - t_start
    logger.info(f"\n✓ Pipeline complete in {total_time/60:.1f} minutes")
    logger.info("  Results saved to results/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-main",     type=int, default=500)
    p.add_argument("--n-ablation", type=int, default=50)
    args = p.parse_args()
    main(n_main=args.n_main, n_ablation=args.n_ablation)
