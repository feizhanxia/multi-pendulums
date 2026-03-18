from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


STRUCTURAL_FEATURES = [
    "fro_norm",
    "asymmetry_norm",
    "asymmetry_ratio",
    "positive_ratio",
    "negative_ratio",
    "mean_abs_weight",
    "std_abs_weight",
    "max_abs_weight",
    "drive_out_strength",
    "drive_in_strength",
    "target_in_strength",
    "target_out_strength",
    "target_sink_bias",
    "target_direct_from_drive",
    "target_path_2hop",
    "target_path_advantage_2hop",
    "target_path_3hop",
    "target_path_advantage_3hop",
]

LABEL_ORDER = ["stable_high", "unstable_peak", "medium", "low"]
LABEL_COLORS = {
    "stable_high": "#007f5f",
    "unstable_peak": "#c44536",
    "medium": "#f4a261",
    "low": "#4d6c91",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#f5f1e8",
            "axes.facecolor": "#fcfaf6",
            "axes.edgecolor": "#2e2a26",
            "axes.labelcolor": "#2e2a26",
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "font.size": 11,
            "grid.color": "#d6cfc2",
            "grid.linestyle": ":",
            "grid.linewidth": 0.8,
            "text.color": "#2e2a26",
            "xtick.color": "#2e2a26",
            "ytick.color": "#2e2a26",
            "savefig.facecolor": "#f5f1e8",
            "savefig.bbox": "tight",
        }
    )


def load_strict_search(json_path: Path) -> dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf-8"))["strict_search"]


def _non_drive_indices(n_nodes: int, drive_index: int) -> list[int]:
    return [i for i in range(n_nodes) if i != drive_index]


def _target_from_amps(amps: np.ndarray, drive_index: int) -> tuple[int, float, float]:
    non_drive = [i for i in range(len(amps)) if i != drive_index]
    nd_amps = amps[non_drive]
    order = np.argsort(nd_amps)
    target_local = int(order[-1])
    second_local = int(order[-2]) if len(order) >= 2 else int(order[-1])
    target_idx = int(non_drive[target_local])
    target_amp = float(nd_amps[target_local])
    second_amp = float(nd_amps[second_local])
    return target_idx, target_amp, second_amp


def _path_scores(K: np.ndarray, drive_index: int) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = K.shape[0]
    direct = np.abs(K[:, drive_index]).astype(float)
    two_hop = np.zeros(n_nodes, dtype=float)
    three_hop = np.zeros(n_nodes, dtype=float)
    for target in range(n_nodes):
        if target == drive_index:
            continue
        for mid in range(n_nodes):
            if mid in {drive_index, target}:
                continue
            two_hop[target] += abs(K[mid, drive_index] * K[target, mid])
        for mid1 in range(n_nodes):
            if mid1 in {drive_index, target}:
                continue
            for mid2 in range(n_nodes):
                if mid2 in {drive_index, target, mid1}:
                    continue
                three_hop[target] += abs(K[mid1, drive_index] * K[mid2, mid1] * K[target, mid2])
    return direct, two_hop, three_hop


def _stage_maps(data: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    robust_map = {int(item["kidx"]): item for item in data.get("robustness_test", [])}
    refined_map = {int(item["kidx"]): item for item in data.get("refined_results", [])}
    final_map = {int(item["kidx"]): item for item in data.get("final_verification", [])}
    return robust_map, refined_map, final_map


def build_candidate_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = data["config"]
    drive_index = int(cfg["drive_index"])
    n_nodes = int(cfg["N"])
    coarse_results = sorted(data["coarse_results"], key=lambda x: x["best_selectivity"], reverse=True)
    coarse_rank = {int(item["kidx"]): rank for rank, item in enumerate(coarse_results, start=1)}
    robust_map, refined_map, final_map = _stage_maps(data)

    coarse_selectivities = np.array([float(item["best_selectivity"]) for item in coarse_results], dtype=float)
    q50, q75, q90 = np.quantile(coarse_selectivities, [0.5, 0.75, 0.9])

    records: list[dict[str, Any]] = []
    for coarse in sorted(data["coarse_results"], key=lambda x: x["kidx"]):
        kidx = int(coarse["kidx"])
        K = np.array(coarse["K"], dtype=float)
        amps = np.array(coarse["best_amps"], dtype=float)
        abs_K = np.abs(K)
        nonzero = int(np.count_nonzero(K))
        pos_edges = int(np.count_nonzero(K > 0))
        neg_edges = int(np.count_nonzero(K < 0))
        target_idx, target_amp, second_amp = _target_from_amps(amps, drive_index)
        direct, two_hop, three_hop = _path_scores(K, drive_index)
        in_strength = abs_K.sum(axis=1)
        out_strength = abs_K.sum(axis=0)
        non_drive = _non_drive_indices(n_nodes, drive_index)
        competitor_2hop = max((two_hop[i] for i in non_drive if i != target_idx), default=0.0)
        competitor_3hop = max((three_hop[i] for i in non_drive if i != target_idx), default=0.0)

        stage_status = "coarse_only"
        if kidx in final_map and bool(final_map[kidx]["stable"]):
            stage_status = "final_stable"
        elif kidx in refined_map:
            stage_status = "refined_failed"
        elif kidx in robust_map:
            stage_status = "robust_failed"

        best_selectivity = float(coarse["best_selectivity"])
        if stage_status == "final_stable":
            broad_label = "stable_high"
        elif kidx in robust_map:
            broad_label = "unstable_peak"
        elif 2.0 <= best_selectivity < 3.0:
            broad_label = "medium"
        else:
            broad_label = "low"

        final_avg = float(final_map[kidx]["avg_selectivity"]) if kidx in final_map else np.nan
        final_min = float(final_map[kidx]["min_selectivity"]) if kidx in final_map else np.nan
        record = {
            "kidx": kidx,
            "coarse_rank": int(coarse_rank[kidx]),
            "best_omega": float(coarse["best_omega"]),
            "best_selectivity": best_selectivity,
            "target_idx": target_idx,
            "target_amp": target_amp,
            "second_amp": second_amp,
            "target_dominance_coarse": target_amp / (second_amp + 1e-12),
            "coarse_target_share": target_amp / (float(np.sum(amps[non_drive])) + 1e-12),
            "stage_status": stage_status,
            "broad_label": broad_label,
            "final_avg_selectivity": final_avg,
            "final_min_selectivity": final_min,
            "fro_norm": float(np.linalg.norm(K)),
            "asymmetry_norm": float(np.linalg.norm(K - K.T)),
            "asymmetry_ratio": float(np.linalg.norm(K - K.T) / (np.linalg.norm(K) + 1e-12)),
            "positive_ratio": float(pos_edges / nonzero) if nonzero else 0.0,
            "negative_ratio": float(neg_edges / nonzero) if nonzero else 0.0,
            "mean_abs_weight": float(abs_K[K != 0].mean()) if nonzero else 0.0,
            "std_abs_weight": float(abs_K[K != 0].std()) if nonzero else 0.0,
            "max_abs_weight": float(abs_K.max()) if nonzero else 0.0,
            "drive_out_strength": float(out_strength[drive_index]),
            "drive_in_strength": float(in_strength[drive_index]),
            "target_in_strength": float(in_strength[target_idx]),
            "target_out_strength": float(out_strength[target_idx]),
            "target_sink_bias": float(in_strength[target_idx] - out_strength[target_idx]),
            "target_direct_from_drive": float(direct[target_idx]),
            "target_path_2hop": float(two_hop[target_idx]),
            "target_path_advantage_2hop": float(two_hop[target_idx] / (competitor_2hop + 1e-12)),
            "target_path_3hop": float(three_hop[target_idx]),
            "target_path_advantage_3hop": float(three_hop[target_idx] / (competitor_3hop + 1e-12)),
        }
        records.append(record)

    return records


def summarize_records(records: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    cfg = data["config"]
    coarse_selectivities = np.array([float(r["best_selectivity"]) for r in records], dtype=float)
    counts = Counter(r["broad_label"] for r in records)
    stage_counts = Counter(r["stage_status"] for r in records)
    stable = [r for r in records if r["broad_label"] == "stable_high"]
    unstable = [r for r in records if r["broad_label"] == "unstable_peak"]
    top_stable = sorted(stable, key=lambda x: x["final_avg_selectivity"], reverse=True)[:5]
    top_unstable = sorted(unstable, key=lambda x: x["best_selectivity"], reverse=True)[:5]

    return {
        "config": cfg,
        "counts": dict(counts),
        "stage_counts": dict(stage_counts),
        "coarse_quantiles": {
            "q50": float(np.quantile(coarse_selectivities, 0.5)),
            "q75": float(np.quantile(coarse_selectivities, 0.75)),
            "q90": float(np.quantile(coarse_selectivities, 0.9)),
            "max": float(np.max(coarse_selectivities)),
        },
        "top_stable": top_stable,
        "top_unstable": top_unstable,
    }


def _feature_scores(records: list[dict[str, Any]]) -> list[tuple[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["broad_label"]].append(record)

    scores: list[tuple[str, float]] = []
    for feature in STRUCTURAL_FEATURES:
        means = []
        for label in LABEL_ORDER:
            rows = grouped.get(label, [])
            if rows:
                means.append(float(np.mean([r[feature] for r in rows])))
        spread = max(means) - min(means) if means else 0.0
        global_std = float(np.std([r[feature] for r in records])) + 1e-12
        scores.append((feature, spread / global_std))
    return sorted(scores, key=lambda x: x[1], reverse=True)


def write_candidate_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) if records else []
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_feature_summary_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for label in LABEL_ORDER:
        label_rows = [r for r in records if r["broad_label"] == label]
        if not label_rows:
            continue
        for feature in STRUCTURAL_FEATURES:
            values = np.array([r[feature] for r in label_rows], dtype=float)
            rows.append(
                {
                    "label": label,
                    "feature": feature,
                    "count": len(label_rows),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "std": float(values.std()),
                }
            )
    fieldnames = ["label", "feature", "count", "mean", "median", "std"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(summary: dict[str, Any], feature_scores: list[tuple[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    payload["feature_scores"] = [{"feature": name, "score": score} for name, score in feature_scores]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_markdown(summary: dict[str, Any], feature_scores: list[tuple[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Analysis Summary",
        "",
        "## Sample Counts",
        "",
    ]
    for label in LABEL_ORDER:
        lines.append(f"- `{label}`: {summary['counts'].get(label, 0)}")
    lines.extend(
        [
            "",
            "## Stage Counts",
            "",
        ]
    )
    for stage, count in sorted(summary["stage_counts"].items()):
        lines.append(f"- `{stage}`: {count}")
    q = summary["coarse_quantiles"]
    lines.extend(
        [
            "",
            "## Coarse Selectivity Quantiles",
            "",
            f"- `q50`: {q['q50']:.3f}",
            f"- `q75`: {q['q75']:.3f}",
            f"- `q90`: {q['q90']:.3f}",
            f"- `max`: {q['max']:.3f}",
            "",
            "## Most Discriminative Structural Features",
            "",
        ]
    )
    for name, score in feature_scores[:8]:
        lines.append(f"- `{name}`: {score:.3f}")

    if summary["top_stable"]:
        lines.extend(["", "## Top Stable Candidates", ""])
        for row in summary["top_stable"]:
            lines.append(
                f"- `K#{row['kidx']}`: omega={row['best_omega']:.2f}, coarse={row['best_selectivity']:.2f}, final_avg={row['final_avg_selectivity']:.2f}"
            )

    if summary["top_unstable"]:
        lines.extend(["", "## Top Unstable Peak Candidates", ""])
        for row in summary["top_unstable"]:
            lines.append(
                f"- `K#{row['kidx']}`: omega={row['best_omega']:.2f}, coarse={row['best_selectivity']:.2f}, stage={row['stage_status']}"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_overview_figure(records: list[dict[str, Any]], feature_scores: list[tuple[str, float]], out_path: Path) -> None:
    _style()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.5), dpi=180, constrained_layout=True)
    ax_hist, ax_counts, ax_scatter, ax_scores = axes.ravel()

    selectivities = np.array([r["best_selectivity"] for r in records], dtype=float)
    ax_hist.hist(selectivities, bins=28, color="#ddb892", edgecolor="#2e2a26", linewidth=0.8)
    ax_hist.set_title("Coarse Selectivity Distribution")
    ax_hist.set_xlabel("Best coarse selectivity")
    ax_hist.set_ylabel("Count (log scale)")
    ax_hist.set_yscale("log")
    for percentile, style in [(0.75, "--"), (0.9, ":")]:
        value = float(np.quantile(selectivities, percentile))
        ax_hist.axvline(value, color="#7f5539", linestyle=style, linewidth=2)

    counts = Counter(r["broad_label"] for r in records)
    labels = [label for label in LABEL_ORDER if counts.get(label, 0) > 0]
    ax_counts.bar(
        labels,
        [counts[label] for label in labels],
        color=[LABEL_COLORS[label] for label in labels],
        edgecolor="#2e2a26",
        linewidth=1.0,
    )
    ax_counts.set_title("Label Counts")
    ax_counts.set_ylabel("Count")
    ax_counts.tick_params(axis="x", rotation=15)
    for idx, label in enumerate(labels):
        value = counts[label]
        ax_counts.text(idx, value, str(value), ha="center", va="bottom", fontsize=10, fontweight="bold")

    non_unstable = [r["best_selectivity"] for r in records if r["broad_label"] != "unstable_peak"]
    scatter_ymax = float(np.quantile(non_unstable, 0.995) * 1.18) if non_unstable else float(np.max(selectivities) * 1.10)
    stable_values = [r["best_selectivity"] for r in records if r["broad_label"] == "stable_high"]
    if stable_values:
        scatter_ymax = max(scatter_ymax, float(max(stable_values) * 1.08))
    scatter_ymax = max(scatter_ymax, 2.0)
    scatter_order = [label for label in ["low", "medium", "unstable_peak", "stable_high"] if label in labels]
    for label in scatter_order:
        rows = [r for r in records if r["broad_label"] == label]
        xs = [r["asymmetry_ratio"] for r in rows]
        ys = np.array([r["best_selectivity"] for r in rows], dtype=float)
        size = 58 if label == "stable_high" else 40
        zorder = 6 if label == "stable_high" else 3
        if label == "unstable_peak":
            within = ys <= scatter_ymax
            over = ~within
            if np.any(within):
                ax_scatter.scatter(
                    np.array(xs)[within],
                    ys[within],
                    s=size,
                    alpha=0.78,
                    color=LABEL_COLORS[label],
                    edgecolors="#2e2a26",
                    linewidths=0.4,
                    label=label,
                    zorder=zorder,
                )
            if np.any(over):
                ax_scatter.scatter(
                    np.array(xs)[over],
                    np.full(int(np.sum(over)), scatter_ymax),
                    s=54,
                    alpha=0.92,
                    marker="^",
                    color=LABEL_COLORS[label],
                    edgecolors="#2e2a26",
                    linewidths=0.5,
                    label=f"{label} (clipped)",
                    zorder=5,
                )
        else:
            ax_scatter.scatter(
                xs,
                ys,
                s=size,
                alpha=0.78,
                color=LABEL_COLORS[label],
                edgecolors="#2e2a26",
                linewidths=0.4,
                label=label,
                zorder=zorder,
            )
    ax_scatter.set_title("Asymmetry vs Coarse Selectivity")
    ax_scatter.set_xlabel(r"$||K-K^T||_F / ||K||_F$")
    ax_scatter.set_ylabel("Best coarse selectivity")
    ax_scatter.set_ylim(0.0, scatter_ymax)
    ax_scatter.legend(frameon=False, fontsize=9)

    top_scores = feature_scores[:8]
    ax_scores.barh(
        [name for name, _ in reversed(top_scores)],
        [score for _, score in reversed(top_scores)],
        color="#588157",
        edgecolor="#2e2a26",
        linewidth=0.8,
    )
    ax_scores.set_title("Most Discriminative Structural Features")
    ax_scores.set_xlabel("normalized mean-gap score")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.6)
    fig.suptitle("N=5 Kij Structure Study: Search and Feature Overview", fontsize=17, fontweight="bold")
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.08, wspace=0.08, hspace=0.08)
    fig.savefig(out_path)
    plt.close(fig)


def _make_feature_profiles(records: list[dict[str, Any]], feature_scores: list[tuple[str, float]], out_path: Path) -> None:
    _style()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top_features = [name for name, _ in feature_scores[:4]]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), dpi=180, constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, feature in zip(axes.ravel(), top_features):
        present = [label for label in LABEL_ORDER if any(r["broad_label"] == label for r in records)]
        for idx, label in enumerate(present):
            values = np.array([r[feature] for r in records if r["broad_label"] == label], dtype=float)
            if values.size == 0:
                continue
            jitter = rng.normal(0.0, 0.045, size=values.size)
            ax.scatter(
                np.full(values.size, idx, dtype=float) + jitter,
                values,
                s=34,
                alpha=0.72,
                color=LABEL_COLORS[label],
                edgecolors="#2e2a26",
                linewidths=0.35,
            )
            ax.hlines(float(values.mean()), idx - 0.25, idx + 0.25, color="#111111", linewidth=2.2)
        ax.set_title(feature.replace("_", " "))
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=18)
        ax.grid(True, axis="y", alpha=0.6)
    fig.suptitle("Feature Profiles by Sample Class", fontsize=17, fontweight="bold")
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.08, wspace=0.08, hspace=0.08)
    fig.savefig(out_path)
    plt.close(fig)


def _make_candidate_gallery(records: list[dict[str, Any]], out_path: Path) -> None:
    _style()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slots: list[tuple[str, dict[str, Any] | None]] = [
        ("Best stable", _pick_record(records, broad_label="stable_high", sort_key="final_avg_selectivity")),
        ("Best unstable", _pick_record(records, broad_label="unstable_peak", sort_key="best_selectivity")),
        ("Best medium", _pick_record(records, broad_label="medium", sort_key="best_selectivity")),
        ("Best low", _pick_record(records, broad_label="low", sort_key="best_selectivity")),
    ]

    fig, axes = plt.subplots(1, len(slots), figsize=(16, 4.8), dpi=180)
    cmap = LinearSegmentedColormap.from_list("earth", ["#355070", "#f8f5ef", "#bc4749"], N=256)
    im = None
    for ax, (title, record) in zip(axes, slots):
        if record is None:
            ax.axis("off")
            continue
        K = np.array(record["K"], dtype=float)
        vmax = float(np.max(np.abs(K))) + 1e-12
        im = ax.imshow(K, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(f"{title}\nK#{record['kidx']}  sel={record['best_selectivity']:.2f}")
        ax.set_xticks(range(K.shape[0]))
        ax.set_yticks(range(K.shape[0]))
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    fig.suptitle("Representative Kij Matrices", fontsize=17, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)


def _pick_record(records: list[dict[str, Any]], broad_label: str, sort_key: str) -> dict[str, Any] | None:
    pool = [r for r in records if r["broad_label"] == broad_label]
    if not pool:
        return None
    def score(row: dict[str, Any]) -> float:
        value = row.get(sort_key, np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return -1e12
        return value if np.isfinite(value) else -1e12

    return max(pool, key=score)


def run_analysis(json_path: Path, out_root: Path) -> dict[str, Any]:
    data = load_strict_search(json_path)
    records = build_candidate_records(data)
    for coarse in data["coarse_results"]:
        lookup = next(r for r in records if r["kidx"] == int(coarse["kidx"]))
        lookup["K"] = coarse["K"]
    summary = summarize_records(records, data)
    feature_scores = _feature_scores(records)

    data_dir = out_root / "data"
    fig_dir = out_root / "figures"
    write_candidate_csv(records, data_dir / "candidate_features.csv")
    write_feature_summary_csv(records, data_dir / "feature_summary.csv")
    write_summary_json(summary, feature_scores, data_dir / "analysis_summary.json")
    write_summary_markdown(summary, feature_scores, out_root / "analysis_summary.md")
    _make_overview_figure(records, feature_scores, fig_dir / "analysis_overview.png")
    _make_feature_profiles(records, feature_scores, fig_dir / "feature_profiles.png")
    _make_candidate_gallery(records, fig_dir / "representative_kij_gallery.png")
    return {"summary": summary, "feature_scores": feature_scores}
