from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_LABELS = {
    "cnn4": "CNN4",
    "rescnn": "ResCNN",
    "ast": "AST",
    "ensemble": "Ensemble",
}
HISTORY_COLUMNS = {
    "epoch",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
}
REPORT_START = "<!-- csv-charts:start -->"
REPORT_END = "<!-- csv-charts:end -->"


def _save_line_chart(
    values: pd.Series,
    epochs: pd.Series,
    title: str,
    ylabel: str,
    destination: Path,
    *,
    ylim: tuple[float, float] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, values, marker="o", linewidth=2)
    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def _save_bar_chart(
    table: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    destination: Path,
    *,
    multiplier: float = 1.0,
    formatter: str = "{:.3g}",
    ylim: tuple[float, float] | None = None,
) -> Path | None:
    values = pd.to_numeric(table[column], errors="raise")
    valid = values.notna()
    if not valid.any():
        return None

    labels = table.loc[valid, "model"].map(
        lambda model: MODEL_LABELS.get(str(model), str(model))
    )
    plotted_values = values.loc[valid] * multiplier
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, plotted_values, color="#168b96")
    ax.set(title=title, ylabel=ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, plotted_values):
        ax.annotate(
            formatter.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def _read_history(path: Path) -> pd.DataFrame:
    history = pd.read_csv(path)
    missing = HISTORY_COLUMNS.difference(history.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    for column in HISTORY_COLUMNS:
        history[column] = pd.to_numeric(history[column], errors="raise")
    if history.empty:
        raise ValueError(f"Training history CSV is empty: {path}")
    return history.sort_values("epoch")


def _read_comparison(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    if "model" not in table.columns:
        raise ValueError(f"{path} is missing the model column")
    if table.empty:
        raise ValueError(f"Comparison CSV is empty: {path}")
    return table


def _plot_histories(root: Path, fold: int, destination: Path) -> list[Path]:
    checkpoint_dir = root / "outputs" / "checkpoints" / f"fold{fold:02d}"
    histories = sorted(checkpoint_dir.glob("*/history.csv"))
    if not histories:
        raise FileNotFoundError(f"No history.csv files found in {checkpoint_dir}")

    generated: list[Path] = []
    all_histories: dict[str, pd.DataFrame] = {}
    for path in histories:
        model = path.parent.name
        history = _read_history(path)
        all_histories[model] = history
        label = MODEL_LABELS.get(model, model)
        for column, title, ylabel, suffix, ylim in (
            ("train_loss", "Training loss", "Loss", "train_loss", None),
            ("val_loss", "Validation loss", "Loss", "val_loss", None),
            ("val_accuracy", "Validation accuracy", "Accuracy", "val_accuracy", (0, 1)),
            ("val_macro_f1", "Validation macro-F1", "Macro-F1", "val_macro_f1", (0, 1)),
        ):
            generated.append(
                _save_line_chart(
                    history[column],
                    history["epoch"],
                    f"{label} — {title} | fold {fold}",
                    ylabel,
                    destination / f"csv_{model}_{suffix}.png",
                    ylim=ylim,
                )
            )

    for column, title, ylabel, suffix, ylim in (
        ("val_loss", "Validation loss — model comparison", "Loss", "val_loss", None),
        ("val_macro_f1", "Validation macro-F1 — model comparison", "Macro-F1", "val_macro_f1", (0, 1)),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        for model, history in all_histories.items():
            label = MODEL_LABELS.get(model, model)
            ax.plot(history["epoch"], history[column], marker="o", label=label)
        ax.set(
            title=f"{title} | fold {fold}",
            xlabel="Epoch",
            ylabel=ylabel,
        )
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = destination / f"csv_comparison_{suffix}_curves.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)
    return generated


def _plot_comparisons(root: Path, fold: int, destination: Path) -> list[Path]:
    metrics_dir = root / "outputs" / "metrics" / f"fold{fold:02d}"
    generated: list[Path] = []

    validation_path = metrics_dir / "validation_comparison.csv"
    if validation_path.is_file():
        validation = _read_comparison(validation_path)
        score_column = (
            "macro_f1" if "macro_f1" in validation.columns else "val_macro_f1"
        )
        if score_column in validation.columns:
            path = _save_bar_chart(
                validation,
                score_column,
                f"Validation macro-F1 by model | fold {fold}",
                "Validation macro-F1",
                destination / "csv_comparison_validation_macro_f1.png",
                ylim=(0, 1),
            )
            if path is not None:
                generated.append(path)

    test_path = metrics_dir / "test_comparison.csv"
    if not test_path.is_file():
        return generated
    test = _read_comparison(test_path)

    for column, title, ylabel, suffix, multiplier, formatter, ylim in (
        ("macro_f1", "Test macro-F1 by model", "Macro-F1", "test_macro_f1", 1, "{:.3f}", (0, 1)),
        ("accuracy", "Test accuracy by model", "Accuracy", "test_accuracy", 1, "{:.3f}", (0, 1)),
        (
            "macro_precision",
            "Test macro-precision by model",
            "Macro-precision",
            "test_macro_precision",
            1,
            "{:.3f}",
            (0, 1),
        ),
        (
            "macro_recall",
            "Test macro-recall by model",
            "Macro-recall",
            "test_macro_recall",
            1,
            "{:.3f}",
            (0, 1),
        ),
        (
            "eval_ms_per_clip_with_loading",
            "Inference throughput by model",
            "Clips per second (1000 / ms per clip)",
            "inference_clips_per_second",
            1000,
            "{:.2f}",
            None,
        ),
        ("checkpoint_mb", "Checkpoint size by model", "Checkpoint size (MB)", "checkpoint_size_mb", 1, "{:.2f}", None),
        ("parameters", "Parameter count by model", "Parameters", "parameters", 1, "{:.3g}", None),
        ("peak_vram_mb", "Peak VRAM by model", "Peak VRAM (MB)", "peak_vram_mb", 1, "{:.1f}", None),
        ("training_seconds", "Training time by model", "Training time (seconds)", "training_seconds", 1, "{:.1f}", None),
    ):
        if column not in test.columns:
            continue
        values = pd.to_numeric(test[column], errors="raise")
        if column == "eval_ms_per_clip_with_loading":
            test = test.copy()
            test[column] = values.where(values > 0)
        path = _save_bar_chart(
            test,
            column,
            f"{title} | fold {fold}",
            ylabel,
            destination / f"csv_comparison_{suffix}.png",
            multiplier=multiplier,
            formatter=formatter,
            ylim=ylim,
        )
        if path is not None:
            generated.append(path)
    return generated


def _update_report(root: Path, fold_charts: list[Path]) -> bool:
    report = root / "outputs" / "report.html"
    if not report.is_file():
        return False

    content = report.read_text(encoding="utf-8")
    if (REPORT_START in content) != (REPORT_END in content):
        raise ValueError(f"Phần biểu đồ CSV trong {report} chưa hoàn chỉnh")

    figures = []
    for chart in fold_charts:
        relative_path = chart.relative_to(root / "outputs").as_posix()
        caption = html.escape(chart.stem.removeprefix("csv_").replace("_", " "))
        figures.append(
            f'<figure><img loading="lazy" src="{html.escape(relative_path, quote=True)}" '
            f'style="max-width:100%"><figcaption>{caption}</figcaption></figure>'
        )
    section = (
        f"{REPORT_START}<section><h2>Biểu đồ tạo từ CSV</h2>"
        + "".join(figures)
        + f"</section>{REPORT_END}"
    )

    if REPORT_START in content:
        content = re.sub(
            re.escape(REPORT_START) + r".*?" + re.escape(REPORT_END),
            lambda _: section,
            content,
            count=1,
            flags=re.DOTALL,
        )
    else:
        closing_body = content.lower().rfind("</body>")
        if closing_body < 0:
            raise ValueError(f"Không tìm thấy thẻ </body> trong {report}")
        content = content[:closing_body] + section + content[closing_body:]
    report.write_text(content, encoding="utf-8")
    return True


def generate_charts(root: Path, folds: list[int] | None = None) -> list[Path]:
    if folds is None:
        checkpoints = root / "outputs" / "checkpoints"
        folds = sorted(
            int(match.group(1))
            for path in checkpoints.glob("fold*/**/history.csv")
            if (match := re.search(r"fold(\d+)", path.as_posix()))
        )
    if not folds:
        raise FileNotFoundError(
            f"No history.csv files found under {root / 'outputs/checkpoints'}"
        )

    all_charts: list[Path] = []
    for fold in folds:
        destination = root / "outputs" / "figures" / f"fold{fold:02d}"
        destination.mkdir(parents=True, exist_ok=True)
        charts = _plot_histories(root, fold, destination)
        charts.extend(_plot_comparisons(root, fold, destination))
        all_charts.extend(charts)
        print(f"Fold {fold}: saved {len(charts)} charts to {destination}")

    if _update_report(root, all_charts):
        print(f"Updated chart gallery in {root / 'outputs' / 'report.html'}")
    return all_charts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot validation, loss, model comparisons, and resource metrics from CSV."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Project directory containing outputs/ (default: project directory).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        action="append",
        help="Fold to plot; may be passed multiple times. Default: discover all folds.",
    )
    args = parser.parse_args()
    generate_charts(args.root.resolve(), args.fold)


if __name__ == "__main__":
    main()
