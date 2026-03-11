import json
from pathlib import Path

import numpy as np

from analyze import main as analyze_main
from scan import main as scan_main
from simulate import main as simulate_main


def test_single_simulation(tmp_path: Path):
    summary_path = tmp_path / "single.csv"
    raw_path = tmp_path / "raw.npz"
    args = [
        "--N",
        "4",
        "--gamma",
        "0.05",
        "--F",
        "0.05",
        "--K",
        "0.2",
        "--Omega",
        "1.0",
        "--t_total",
        "50",
        "--dt",
        "1.0",
        "--discard_ratio",
        "0.4",
        "--seed",
        "0",
        "--save_raw",
        "--raw_path",
        str(raw_path),
        "--summary_path",
        str(summary_path),
    ]
    simulate_main(args)
    assert summary_path.exists()
    assert raw_path.exists()
    data = np.load(raw_path)
    assert "theta" in data
    summary = summary_path.read_text().strip().splitlines()
    assert len(summary) == 2  # header + one record


def test_scan_and_analyze(tmp_path: Path):
    summary_path = tmp_path / "grid.csv"
    raw_dir = tmp_path / "raw"
    figures_dir = tmp_path / "figs"

    scan_args = [
        "--N",
        "4",
        "--gamma",
        "0.05",
        "--F",
        "0.05",
        "--w0",
        "1.0",
        "--t_total",
        "40",
        "--dt",
        "1.0",
        "--discard_ratio",
        "0.4",
        "--seed",
        "0",
        "--drive_index",
        "0",
        "--omega_range",
        "0.8",
        "1.0",
        "0.2",
        "--k_range",
        "0.1",
        "0.2",
        "0.1",
        "--workers",
        "1",
        "--save_raw",
        "--raw_dir",
        str(raw_dir),
        "--summary_path",
        str(summary_path),
    ]
    scan_main(scan_args)
    assert summary_path.exists()
    raw_files = list(raw_dir.glob("*.npz"))
    assert raw_files

    analyze_args = [
        "--summary_path",
        str(summary_path),
        "--metric",
        "selectivity",
        "--top",
        "2",
        "--out_dir",
        str(figures_dir),
    ]
    analyze_main(analyze_args)
    heatmap = figures_dir / "heatmap_selectivity.png"
    assert heatmap.exists()
    bars = list(figures_dir.glob("top*.png"))
    assert bars
