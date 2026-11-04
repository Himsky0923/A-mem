"""Aggregate test_advanced_robust.py results over trials: mean ± std per condition.

A condition is model x evolution on/off x retrieve_k x ratio; its trials are the result files that share them
(usually one per seed). Writes a Markdown report and CSV tables:

    python scripts/aggregate.py results/*.json --out results/summary

- scores_<scoring>.csv: F1 and BLEU-1 (%) per category and overall, for both category-5 scorings
  ('legacy' = against the adversarial answer, as in the paper; 'not_mentioned' = corrected)
- cost.csv: LLM calls, tokens, time, failed / overflowing answers, parse fallbacks, neighbour updates

Standard library only.
"""
import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

CATEGORIES = ["multi_hop", "temporal", "open_domain", "single_hop", "adversarial"]
NAMES = {"multi_hop": "Multi-hop", "temporal": "Temporal", "open_domain": "Open-domain",
         "single_hop": "Single-hop", "adversarial": "Adversarial", "overall": "Overall"}
SCORINGS = ["legacy", "not_mentioned"]

# A-Mem results reported in the paper (arXiv 2502.12110, latest version), F1 / BLEU-1 in %.
# Category 5 there is scored against the adversarial answer ('legacy').
PAPER = {
    "Qwen2.5-3B, A-Mem (Table 1)": {"multi_hop": (12.57, 9.01), "temporal": (27.59, 25.07),
                                     "open_domain": (7.12, 7.28), "single_hop": (17.23, 13.12),
                                     "adversarial": (27.91, 25.15)},
    "GPT-4o-mini, A-Mem (Table 1)": {"multi_hop": (27.02, 20.09), "temporal": (45.85, 36.67),
                                      "open_domain": (12.14, 12.00), "single_hop": (44.65, 37.06),
                                      "adversarial": (50.03, 49.47)},
    "GPT-4o-mini, w/o memory evolution (Table 3)": {"multi_hop": (21.35, 15.13), "temporal": (31.24, 27.31),
                                                    "open_domain": (10.13, 10.85), "single_hop": (39.17, 34.70),
                                                    "adversarial": (44.16, 45.33)},
}


def condition_of(run):
    s = run.get("settings", {})
    k = s.get("retrieve_k_by_category")
    k_label = "k=" + ("/".join(str(k[c]) for c in sorted(k)) if k else str(s.get("retrieve_k", "?")))
    evolution = run.get("evolution", s.get("evolution", True))
    return (run.get("model", "?"), "evolution" if evolution else "no evolution", k_label, f"ratio={run.get('ratio')}")


def scores_of(run, scoring):
    """{category name or 'overall': (f1, bleu1)} in %, for one run and one category-5 scoring."""
    aggregate = run.get("aggregate_metrics_by_c5_scoring", {}).get(scoring)
    if aggregate is None:  # results written before both scorings were saved
        main = "legacy" if run.get("legacy_c5_reference") else "not_mentioned"
        aggregate = run["aggregate_metrics"] if scoring == main else None
    if aggregate is None:
        return {}
    names = {f"category_{i}": c for i, c in enumerate(CATEGORIES, start=1)}
    out = {}
    for split, metrics in aggregate.items():
        name = "overall" if split == "overall" else names.get(split)
        if name and "f1" in metrics:
            out[name] = (100 * metrics["f1"]["mean"], 100 * metrics.get("bleu1", {}).get("mean", 0.0))
    return out


def cost_of(run):
    cost, failures = run.get("cost", {}), run.get("llm_failures", {})
    built, qa = cost.get("construction", {}), cost.get("qa", {})
    questions = run.get("total_questions") or 0
    row = {
        "qa_calls": qa.get("calls"), "qa_tokens": (qa.get("prompt_tokens", 0) + qa.get("completion_tokens", 0))
        if qa else None, "qa_minutes": qa.get("seconds", 0) / 60 if qa else None,
        "max_qa_prompt_tokens": qa.get("max_prompt_tokens"),
        "answer_failures": failures.get("answer", 0),
        "context_overflows": failures.get("answer_context_overflow", 0),
        "overflow_pct": 100 * failures.get("answer_context_overflow", 0) / questions if questions else None,
        "other_llm_failures": sum(v for k, v in failures.items() if not k.startswith("answer")),
        "parse_fallbacks": sum(run.get("parse_fallbacks", {}).values()),
        "neighbor_notes_updated": run.get("evolution_stats", {}).get("neighbor_notes_updated", 0),
        "neighbor_updates_skipped": run.get("evolution_stats", {}).get("neighbor_update_skipped", 0),
    }
    # memory construction is only measured in runs that built the memories (not loaded them from the cache)
    if built.get("samples_built"):
        row.update({"build_calls": built.get("calls"),
                    "build_tokens": built.get("prompt_tokens", 0) + built.get("completion_tokens", 0),
                    "build_minutes": built.get("seconds", 0) / 60})
    return row


def mean_std(values, digits=2):
    values = [v for v in values if v is not None]
    if not values:
        return "–", None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    text = f"{mean:.{digits}f}" + (f" ± {std:.{digits}f}" if std is not None else "")
    return text, mean, std


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="results JSON files or glob patterns")
    parser.add_argument("--out", default="results/summary", help="output prefix (writes <out>.md and CSVs)")
    args = parser.parse_args()

    files = sorted({f for pattern in args.results for f in (glob.glob(pattern) or [pattern])})
    groups = defaultdict(list)
    for path in files:
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        if "aggregate_metrics" not in run:
            print(f"skipping {path}: not a results file")
            continue
        run["_file"] = os.path.basename(path)
        groups[condition_of(run)].append(run)
    if not groups:
        raise SystemExit("no results files found")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    md = ["# A-Mem on LoCoMo: mean ± std over trials", "",
          "F1 and BLEU-1 in %. ± is the sample standard deviation over the trials (seeds) of a condition.", ""]

    md += ["## Conditions", "", "| Model | Evolution | k | Ratio | Trials | Seeds | Files |", "|---|---|---|---|---|---|---|"]
    for cond, runs in sorted(groups.items()):
        seeds = ", ".join(str(r.get("seed")) for r in runs)
        md.append(f"| {' | '.join(cond)} | {len(runs)} | {seeds} | {', '.join(r['_file'] for r in runs)} |")
    md.append("")

    columns = CATEGORIES + ["overall"]
    for scoring in SCORINGS:
        title = ("category 5 scored against the adversarial answer, as in the paper" if scoring == "legacy"
                 else "category 5 scored against 'Not mentioned in the conversation' (corrected)")
        md += [f"## Scores ({scoring}: {title})", "",
               "| Model | Evolution | k | Ratio | " + " | ".join(f"{NAMES[c]} F1 | {NAMES[c]} BLEU-1" for c in columns) + " |",
               "|" + "---|" * (4 + 2 * len(columns))]
        rows = []
        for cond, runs in sorted(groups.items()):
            per_run = [scores_of(r, scoring) for r in runs]
            cells, row = [], dict(zip(["model", "evolution", "k", "ratio"], cond), trials=len(runs))
            for c in columns:
                for i, metric in enumerate(("f1", "bleu1")):
                    text, mean, std = mean_std([s[c][i] for s in per_run if c in s])
                    cells.append(text)
                    row[f"{c}_{metric}_mean"], row[f"{c}_{metric}_std"] = mean, std
            md.append(f"| {' | '.join(cond)} | " + " | ".join(cells) + " |")
            rows.append(row)
        md.append("")
        with open(f"{args.out}_scores_{scoring}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    md += ["## Paper (A-Mem, F1 / BLEU-1 %, category 5 scored as 'legacy')", "",
           "| Source | " + " | ".join(NAMES[c] for c in CATEGORIES) + " |", "|---|" + "---|" * len(CATEGORIES)]
    for source, scores in PAPER.items():
        md.append(f"| {source} | " + " | ".join(f"{scores[c][0]:.2f} / {scores[c][1]:.2f}" for c in CATEGORIES) + " |")
    md.append("")

    cost_fields = [("build_calls", "Build LLM calls", 0), ("build_tokens", "Build tokens", 0),
                   ("build_minutes", "Build min", 1), ("qa_calls", "QA LLM calls", 0), ("qa_tokens", "QA tokens", 0),
                   ("qa_minutes", "QA min", 1), ("max_qa_prompt_tokens", "Max QA prompt tokens", 0),
                   ("answer_failures", "Failed answers", 1), ("context_overflows", "Context overflows", 1),
                   ("overflow_pct", "Overflow % of questions", 2), ("other_llm_failures", "Other LLM failures", 1),
                   ("parse_fallbacks", "Parse fallbacks", 0), ("neighbor_notes_updated", "Neighbour notes rewritten", 0),
                   ("neighbor_updates_skipped", "Neighbour updates skipped", 0)]
    md += ["## Cost and reliability (mean ± std over trials)", "",
           "Build columns only count trials that built their memories (not loaded them from the cache). "
           "Failed answers are scored as empty predictions; context overflows are the subset rejected for "
           "exceeding the context window.", "",
           "| Model | Evolution | k | Ratio | " + " | ".join(label for _, label, _ in cost_fields) + " |",
           "|" + "---|" * (4 + len(cost_fields))]
    cost_rows = []
    for cond, runs in sorted(groups.items()):
        per_run = [cost_of(r) for r in runs]
        cells, row = [], dict(zip(["model", "evolution", "k", "ratio"], cond), trials=len(runs))
        for key, _, digits in cost_fields:
            text, mean, std = mean_std([c.get(key) for c in per_run], digits)
            cells.append(text)
            row[f"{key}_mean"], row[f"{key}_std"] = mean, std
        md.append(f"| {' | '.join(cond)} | " + " | ".join(cells) + " |")
        cost_rows.append(row)
    md.append("")
    with open(f"{args.out}_cost.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cost_rows[0]))
        writer.writeheader()
        writer.writerows(cost_rows)

    with open(f"{args.out}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {args.out}.md, {args.out}_scores_legacy.csv, {args.out}_scores_not_mentioned.csv, "
          f"{args.out}_cost.csv")


if __name__ == "__main__":
    main()
