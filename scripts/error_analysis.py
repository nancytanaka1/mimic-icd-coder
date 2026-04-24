"""Baseline error analysis — extract per-label metrics, calibration, and
confusion pairs from the shipped TF-IDF + LR baseline.

Non-training. Reads existing artifacts from data/silver, data/gold, and
data/bronze; runs the saved model against the held-out test split;
computes per-label F1/precision/recall/support/threshold/Brier, a
reliability curve, and pair-wise false-positive substitution counts.

Outputs (all under reports/):
    baseline_per_label.parquet      50-row table for the error-analysis doc
    baseline_confusion_pairs.parquet top-N confused label pairs
    figures/baseline_calibration.png reliability curve + Brier distribution
    figures/baseline_confusion_pairs.png 10x10 confusion heatmap
    stdout                           markdown table + top-10 lists, ready to
                                     paste into reports/baseline_error_analysis.md

Usage:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --log-level INFO
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support

from mimic_icd_coder.logging_utils import configure_logging, get_logger
from mimic_icd_coder.models.baseline import BaselineModel

logger = get_logger(__name__)


def _df_to_markdown(df: pd.DataFrame, float_cols: tuple[str, ...] = ()) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table.

    ``float_cols`` are formatted to 3 decimals; everything else is ``str(...)``.
    """
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in float_cols and isinstance(v, int | float | np.floating):
                cells.append(f"{float(v):.3f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifacts", default="data", help="Data root (default: data)")
    parser.add_argument("--reports", default="reports", help="Reports output dir")
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity (default: DEBUG so progress is visible)",
    )
    parser.add_argument(
        "--top-n-confused", type=int, default=10, help="How many confusion pairs to emit"
    )
    args = parser.parse_args()

    configure_logging(level=args.log_level)

    artifacts = Path(args.artifacts)
    reports = Path(args.reports)
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Load artifacts
    # ----------------------------------------------------------------------
    logger.info("error_analysis.load.start", artifacts=str(artifacts))

    silver = pd.read_parquet(artifacts / "silver" / "notes.parquet")
    y = load_npz(artifacts / "gold" / "labels.npz")
    label_names: list[str] = json.loads(
        (artifacts / "gold" / "label_names.json").read_text(encoding="utf-8")
    )
    splits = pd.read_parquet(artifacts / "gold" / "splits.parquet")
    test_idx = splits.loc[splits["split"] == "test", "row_idx"].to_numpy()
    logger.debug(
        "error_analysis.loaded.splits",
        n_silver=len(silver),
        n_labels=y.shape[1],
        n_test=len(test_idx),
    )

    model = BaselineModel.load(artifacts / "gold" / "baseline_model.joblib")
    thresholds = np.load(artifacts / "gold" / "baseline_thresholds.npy")
    logger.debug(
        "error_analysis.loaded.model",
        threshold_shape=thresholds.shape,
        thr_min=float(thresholds.min()),
        thr_median=float(np.median(thresholds)),
        thr_max=float(thresholds.max()),
    )

    d_icd_path = artifacts / "bronze" / "d_icd_diagnoses.parquet"
    if d_icd_path.is_file():
        d_icd = pd.read_parquet(d_icd_path)
        d_icd_10 = (
            d_icd.loc[d_icd["icd_version"] == 10]
            .drop_duplicates(subset=["icd_code"])
            .set_index("icd_code")["long_title"]
        )
        logger.debug("error_analysis.loaded.icd_dict", n_icd10_codes=len(d_icd_10))
    else:
        logger.warning("error_analysis.no_icd_dict", path=str(d_icd_path))
        d_icd_10 = pd.Series(dtype=str)

    # ----------------------------------------------------------------------
    # Predict on test split
    # ----------------------------------------------------------------------
    logger.info("error_analysis.predict.start", n_test=len(test_idx))
    test_texts = [silver["text"].iloc[i] for i in test_idx]
    y_test = np.asarray(y[test_idx].todense(), dtype=np.int8)
    y_prob = model.predict_proba(test_texts)
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int8)
    logger.info(
        "error_analysis.predict.done",
        y_prob_shape=y_prob.shape,
        positives_true_total=int(y_test.sum()),
        positives_pred_total=int(y_pred.sum()),
        mean_pred_rate=float(y_pred.mean()),
    )

    # ----------------------------------------------------------------------
    # Per-label metrics
    # ----------------------------------------------------------------------
    logger.info("error_analysis.per_label.start")
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    briers = np.array(
        [brier_score_loss(y_test[:, j], y_prob[:, j]) for j in range(len(label_names))]
    )
    logger.debug(
        "error_analysis.per_label.computed",
        macro_f1=float(f1s.mean()),
        min_support=int(supports.min()),
        max_support=int(supports.max()),
        macro_brier=float(briers.mean()),
    )

    per_label = pd.DataFrame(
        {
            "code": label_names,
            "description": [d_icd_10.get(c, "(no description)") for c in label_names],
            "support": supports.astype(int),
            "precision": precisions,
            "recall": recalls,
            "f1": f1s,
            "threshold": thresholds,
            "brier": briers,
        }
    )
    per_label_path = reports / "baseline_per_label.parquet"
    per_label.to_parquet(per_label_path, index=False)
    logger.info("error_analysis.per_label.saved", path=str(per_label_path), rows=len(per_label))

    # ----------------------------------------------------------------------
    # Calibration: reliability curve (aggregated) + per-label Brier histogram
    # ----------------------------------------------------------------------
    logger.info("error_analysis.calibration.start")
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    flat_prob = y_prob.reshape(-1)
    flat_true = y_test.reshape(-1)
    bin_pred_mean, bin_obs_rate, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (flat_prob >= bin_edges[i]) & (
            (flat_prob < bin_edges[i + 1]) if i < n_bins - 1 else (flat_prob <= bin_edges[i + 1])
        )
        if mask.sum() > 0:
            bin_pred_mean.append(float(flat_prob[mask].mean()))
            bin_obs_rate.append(float(flat_true[mask].mean()))
            bin_counts.append(int(mask.sum()))
            logger.debug(
                "error_analysis.calibration.bin",
                bin=i,
                edge_lo=float(bin_edges[i]),
                edge_hi=float(bin_edges[i + 1]),
                n=int(mask.sum()),
                pred_mean=float(flat_prob[mask].mean()),
                obs_rate=float(flat_true[mask].mean()),
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax1.plot(bin_pred_mean, bin_obs_rate, "o-", color="steelblue", label="Baseline")
    ax1.set_xlabel("Mean predicted probability (binned)")
    ax1.set_ylabel("Observed positive rate")
    ax1.set_title(f"Reliability curve — aggregated across {len(label_names)} labels")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.hist(briers, bins=20, edgecolor="black", color="steelblue", alpha=0.7)
    ax2.axvline(
        float(briers.mean()),
        color="red",
        linestyle="--",
        label=f"Macro Brier = {briers.mean():.3f}",
    )
    ax2.set_xlabel("Brier score")
    ax2.set_ylabel("# labels")
    ax2.set_title("Per-label Brier-score distribution")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    calibration_path = figures / "baseline_calibration.png"
    plt.savefig(calibration_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(
        "error_analysis.calibration.saved",
        path=str(calibration_path),
        macro_brier=float(briers.mean()),
    )

    # ----------------------------------------------------------------------
    # Confusion pair analysis
    # false_sub[A, B] = # test admissions where A is in truth AND B is a
    # false positive (B predicted, B not in truth). Captures sibling-code
    # confusion patterns.
    # ----------------------------------------------------------------------
    logger.info("error_analysis.confusion.start")
    n_labels = len(label_names)
    false_sub = np.zeros((n_labels, n_labels), dtype=np.int64)
    fp_mask = (y_pred == 1) & (y_test == 0)  # every false-positive cell
    for a in range(n_labels):
        true_a = y_test[:, a] == 1
        if not true_a.any():
            continue
        # For admissions where A is truly present, count FPs on every label B.
        false_sub[a, :] = fp_mask[true_a].sum(axis=0)
    np.fill_diagonal(false_sub, 0)
    logger.debug(
        "error_analysis.confusion.matrix_built",
        shape=false_sub.shape,
        total_pairs=int((false_sub > 0).sum()),
        max_count=int(false_sub.max()),
    )

    pairs = []
    for i in range(n_labels):
        for j in range(n_labels):
            if i == j or false_sub[i, j] == 0:
                continue
            support_i = int(y_test[:, i].sum())
            pairs.append(
                {
                    "true_code": label_names[i],
                    "true_desc": str(d_icd_10.get(label_names[i], "(no description)")),
                    "wrongly_predicted_code": label_names[j],
                    "wrongly_predicted_desc": str(d_icd_10.get(label_names[j], "(no description)")),
                    "count": int(false_sub[i, j]),
                    "true_support": support_i,
                    "rate_per_true": float(false_sub[i, j] / support_i) if support_i > 0 else 0.0,
                }
            )
    pairs_df = (
        pd.DataFrame(pairs)
        .sort_values("count", ascending=False)
        .head(args.top_n_confused)
        .reset_index(drop=True)
    )
    pairs_path = reports / "baseline_confusion_pairs.parquet"
    pairs_df.to_parquet(pairs_path, index=False)
    logger.info("error_analysis.confusion.saved", path=str(pairs_path), rows=len(pairs_df))

    # Confusion heatmap — use the 10 labels most involved in confusion pairs
    # (rows + columns of false_sub summed).
    total_involvement = false_sub.sum(axis=1) + false_sub.sum(axis=0)
    top_confused = np.argsort(total_involvement)[::-1][:10]
    heat = false_sub[np.ix_(top_confused, top_confused)]

    fig2, ax3 = plt.subplots(figsize=(9, 7))
    im = ax3.imshow(heat, cmap="Reds")
    tick_labels = [label_names[i] for i in top_confused]
    ax3.set_xticks(range(10))
    ax3.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax3.set_yticks(range(10))
    ax3.set_yticklabels(tick_labels)
    ax3.set_xlabel("Wrongly predicted (false positive)")
    ax3.set_ylabel("True label")
    ax3.set_title("Top-10 most-confused labels — false positives of column given true row")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    thresh = heat.max() / 2 if heat.max() > 0 else 1
    for i in range(10):
        for j in range(10):
            if heat[i, j] > 0:
                ax3.text(
                    j,
                    i,
                    str(int(heat[i, j])),
                    ha="center",
                    va="center",
                    color="white" if heat[i, j] > thresh else "black",
                    fontsize=8,
                )
    plt.tight_layout()
    confusion_path = figures / "baseline_confusion_pairs.png"
    plt.savefig(confusion_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("error_analysis.confusion_heatmap.saved", path=str(confusion_path))

    # ----------------------------------------------------------------------
    # Stdout summaries — ready to paste into the prose doc
    # ----------------------------------------------------------------------
    worst10 = per_label.sort_values("f1").head(10).reset_index(drop=True)
    best10 = per_label.sort_values("f1", ascending=False).head(10).reset_index(drop=True)

    print("\n" + "=" * 72)
    print("  TOP 10 WORST LABELS (ascending F1)")
    print("=" * 72)
    for _, r in worst10.iterrows():
        desc = r["description"][:60] + ("..." if len(r["description"]) > 60 else "")
        print(
            f"  {r['code']:<8s} F1={r['f1']:.3f}  support={int(r['support']):>5d}  "
            f"P={r['precision']:.3f} R={r['recall']:.3f}  {desc}"
        )

    print("\n" + "=" * 72)
    print("  TOP 10 BEST LABELS (F1 descending)")
    print("=" * 72)
    for _, r in best10.iterrows():
        desc = r["description"][:60] + ("..." if len(r["description"]) > 60 else "")
        print(
            f"  {r['code']:<8s} F1={r['f1']:.3f}  support={int(r['support']):>5d}  "
            f"P={r['precision']:.3f} R={r['recall']:.3f}  {desc}"
        )

    print("\n" + "=" * 72)
    print(f"  TOP {len(pairs_df)} CONFUSION PAIRS (FP of B when A is true)")
    print("=" * 72)
    for _, r in pairs_df.iterrows():
        td = r["true_desc"][:35] + ("..." if len(r["true_desc"]) > 35 else "")
        wd = r["wrongly_predicted_desc"][:35] + (
            "..." if len(r["wrongly_predicted_desc"]) > 35 else ""
        )
        print(
            f"  true {r['true_code']:<8s} ({td:<38}) "
            f"-> wrongly pred {r['wrongly_predicted_code']:<8s} ({wd}): "
            f"{r['count']:>4d}x  ({r['rate_per_true']:.2%} of true cases)"
        )

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Per-label table:         {per_label_path}  ({len(per_label)} rows)")
    print(f"  Calibration plot:        {calibration_path}")
    print(f"  Confusion heatmap:       {confusion_path}")
    print(f"  Confusion pairs table:   {pairs_path}  (top {len(pairs_df)} pairs)")
    print(f"  Macro Brier score:       {float(briers.mean()):.4f}")
    print(
        f"  F1 range:                worst {worst10['f1'].min():.3f} ({worst10.iloc[0]['code']})"
        f"  ...  best {best10['f1'].max():.3f} ({best10.iloc[0]['code']})"
    )

    # Emit the 50-row per_label markdown table for the prose doc — printed at
    # the end so a reviewer tailing the log sees it last.
    md_cols = per_label[
        ["code", "description", "support", "precision", "recall", "f1", "threshold"]
    ].sort_values("f1")
    # Round floats for readability.
    md_cols = md_cols.assign(
        precision=md_cols["precision"].round(3),
        recall=md_cols["recall"].round(3),
        f1=md_cols["f1"].round(3),
        threshold=md_cols["threshold"].round(3),
    )
    md_path = reports / "baseline_per_label_table.md"
    md_path.write_text(
        _df_to_markdown(md_cols, float_cols=("precision", "recall", "f1", "threshold")) + "\n",
        encoding="utf-8",
    )
    logger.info("error_analysis.markdown_table.saved", path=str(md_path))

    print(f"\n  Markdown table (paste into prose): {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
