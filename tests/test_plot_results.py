from pathlib import Path

import pandas as pd
import pytest

from scripts.plot_results import generate_charts


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def _make_results(root: Path) -> None:
    histories = pd.DataFrame(
        {
            "epoch": [1, 2],
            "train_loss": [1.0, 0.5],
            "train_accuracy": [0.5, 0.8],
            "val_loss": [0.8, 0.4],
            "val_accuracy": [0.6, 0.85],
            "val_macro_f1": [0.55, 0.82],
        }
    )
    for model in ("cnn4", "ast"):
        _write_csv(
            root / "outputs" / "checkpoints" / "fold10" / model / "history.csv",
            histories,
        )

    _write_csv(
        root / "outputs" / "metrics" / "fold10" / "validation_comparison.csv",
        pd.DataFrame(
            {"model": ["ast", "cnn4"], "macro_f1": [0.82, 0.7]}
        ),
    )
    _write_csv(
        root / "outputs" / "metrics" / "fold10" / "test_comparison.csv",
        pd.DataFrame(
            {
                "model": ["ast", "cnn4", "ensemble"],
                "macro_f1": [0.8, 0.7, 0.81],
                "accuracy": [0.82, 0.71, 0.83],
                "macro_precision": [0.84, 0.73, 0.85],
                "macro_recall": [0.81, 0.72, 0.82],
                "eval_ms_per_clip_with_loading": [20.0, 2.0, float("nan")],
                "checkpoint_mb": [300.0, 5.0, 305.0],
                "parameters": [80_000_000, 1_000_000, 81_000_000],
                "peak_vram_mb": [1600.0, 400.0, float("nan")],
                "training_seconds": [9000.0, 500.0, 9500.0],
            }
        ),
    )
    report = root / "outputs" / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("<html><body><table></table></body></html>", encoding="utf-8")


def test_generate_charts_separates_history_metrics_and_updates_report(tmp_path):
    _make_results(tmp_path)

    charts = generate_charts(tmp_path)

    names = {chart.name for chart in charts}
    assert "csv_cnn4_train_loss.png" in names
    assert "csv_cnn4_val_loss.png" in names
    assert "csv_cnn4_val_accuracy.png" in names
    assert "csv_cnn4_val_macro_f1.png" in names
    assert "csv_comparison_val_loss_curves.png" in names
    assert "csv_comparison_val_macro_f1_curves.png" in names
    assert "csv_comparison_test_macro_precision.png" in names
    assert "csv_comparison_test_macro_recall.png" in names
    assert "csv_comparison_inference_clips_per_second.png" in names
    assert "csv_comparison_checkpoint_size_mb.png" in names
    assert all(chart.is_file() and chart.stat().st_size > 0 for chart in charts)

    report = (tmp_path / "outputs" / "report.html").read_text(encoding="utf-8")
    assert report.count("<!-- csv-charts:start -->") == 1
    assert "csv_comparison_inference_clips_per_second.png" in report


def test_generate_charts_replaces_existing_report_gallery(tmp_path):
    _make_results(tmp_path)

    generate_charts(tmp_path, [10])
    generate_charts(tmp_path, [10])

    report = (tmp_path / "outputs" / "report.html").read_text(encoding="utf-8")
    assert report.count("<!-- csv-charts:start -->") == 1
    assert report.count("csv_cnn4_val_loss.png") == 1


def test_generate_charts_reports_missing_history(tmp_path):
    with pytest.raises(FileNotFoundError, match="history.csv"):
        generate_charts(tmp_path)
