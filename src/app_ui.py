from __future__ import annotations

import json
import html
from datetime import datetime
import math
import time
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .engine import (
    CLASSES, MODEL_NAMES, VI, available_checkpoint_kinds, load_audio, logmel,
    predict_file, warmup_prediction_model,
)

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
DEVICE_CHOICES = ["CPU"] + (["CUDA (GPU)"] if torch.cuda.is_available() else [])


class AudioInputError(ValueError):
    """User-facing audio validation error."""


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu tệp cấu hình cần thiết: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def available_models(root: Path = ROOT) -> list[str]:
    config = _read_json(root / "configs" / "config.json")
    fold = config["test_folds"][0]
    models = available_checkpoint_kinds(root, fold, config)
    if "ast" in models:
        if not all(
            (root / path).exists()
            for path in (
                "configs/ast_7class/config.json",
                "configs/ast_processor/preprocessor_config.json",
            )
        ):
            models.remove("ast")

    ensemble_path = root / "outputs" / "metrics" / f"fold{fold:02d}" / "ensemble.json"
    if ensemble_path.is_file():
        ensemble = _read_json(ensemble_path)
        active_models = [k for k, weight in zip(ensemble["kinds"], ensemble["weights"]) if weight > 0]
        if active_models and all(kind in models for kind in active_models):
            models.append("ensemble")
    elif len(models) > 1:
        models.append("ensemble")
    return models


def validate_audio(path: str | Path, root: Path = ROOT) -> tuple[np.ndarray, int, float]:
    if not path:
        raise AudioInputError("Hãy chọn tệp âm thanh hoặc ghi âm bằng microphone.")
    audio_path = Path(path)
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioInputError("Định dạng chưa hỗ trợ. Hãy dùng WAV, FLAC, OGG hoặc MP3.")
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise AudioInputError("Tệp âm thanh không tồn tại hoặc rỗng.")

    app_config = _read_json(root / "configs" / "app_config.json")
    max_duration = float(app_config["max_duration_seconds"])
    try:
        info = sf.info(str(audio_path))
        duration = float(info.duration)
        if info.frames <= 0 or duration <= 0:
            raise AudioInputError("Tệp âm thanh không có mẫu âm thanh.")
        if not math.isfinite(duration) or duration > max_duration:
            raise AudioInputError(
                f"Thời lượng không hợp lệ hoặc quá dài (tối đa {max_duration:g} giây)."
            )
        waveform = load_audio(audio_path, sr=16000, seconds=None)
    except AudioInputError:
        raise
    except (OSError, RuntimeError) as exc:
        raise AudioInputError(
            "Không đọc được âm thanh. Tệp có thể bị hỏng hoặc codec chưa được hỗ trợ; "
            "hãy thử WAV/FLAC hoặc chuyển mã tệp."
        ) from exc
    except ValueError as exc:
        message = str(exc)
        if "NaN/Inf" in message:
            raise AudioInputError("Âm thanh chứa giá trị NaN/Inf, không thể dự đoán.") from exc
        raise AudioInputError("Tệp âm thanh không hợp lệ hoặc bị hỏng.") from exc

    if waveform.size == 0:
        raise AudioInputError("Tệp âm thanh không có mẫu âm thanh.")
    if not np.isfinite(waveform).all():
        raise AudioInputError("Âm thanh chứa giá trị NaN/Inf, không thể dự đoán.")
    cfg = _read_json(root / "configs" / "config.json")
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    if rms < float(cfg["silence_rms"]):
        raise AudioInputError(
            "Âm thanh im lặng hoặc quá nhỏ. Hãy ghi/chọn âm thanh rõ hơn rồi thử lại."
        )
    return waveform, 16000, duration


def _make_waveform_plot(waveform: np.ndarray, sr: int):
    stride = max(1, math.ceil(len(waveform) / 25000))
    times = np.arange(0, len(waveform), stride) / sr
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, waveform[::stride], linewidth=0.6, color="#147d83")
    ax.set(
        title="Waveform",
        xlabel="Thời gian (giây)",
        ylabel="Biên độ",
        xlim=(0, len(waveform) / sr),
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def _make_logmel_plot(waveform: np.ndarray, sr: int, root: Path = ROOT):
    config = _read_json(root / "configs" / "config.json")
    window_samples = int(config["seconds"] * sr)
    window = waveform[:window_samples]
    if len(window) < window_samples:
        window = np.pad(window, (0, window_samples - len(window)))
    mel = logmel(window, config)
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    ax.set(title="Log-Mel spectrogram — cửa sổ đầu 4 giây", xlabel="Khung thời gian", ylabel="Mel bin")
    fig.colorbar(image, ax=ax, label="dB")
    fig.tight_layout()
    return fig


def _metric_table(root: Path = ROOT):
    config = _read_json(root / "configs" / "config.json")
    fold = config["test_folds"][0]
    path = root / "outputs" / "metrics" / f"fold{fold:02d}" / "test_comparison.csv"
    if not path.is_file():
        return None, f"Chưa tìm thấy metrics test fold {fold}: {path.relative_to(root)}."
    table = pd.read_csv(path)
    wanted = [
        "model", "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "val_macro_f1", "selected_by_validation",
    ]
    return table[[column for column in wanted if column in table.columns]], (
        f"Số liệu đọc từ {path.relative_to(root)} — test fold {fold} của lần chạy hiện tại. "
        "Đây không phải độ chính xác tổng quát."
    )


def predict_audio(path, model_choice: str, device_choice: str, threshold: float, root: Path = ROOT):
    try:
        request_started = time.perf_counter()
        waveform, sample_rate, duration = validate_audio(path, root)
        models = available_models(root)
        if model_choice not in models:
            raise FileNotFoundError(
                f"Không tìm thấy đầy đủ tệp của mô hình {model_choice!r}. "
                "Kiểm tra outputs/checkpoints và configs."
            )
        if device_choice == "CUDA (GPU)":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA không khả dụng; chọn CPU hoặc kiểm tra driver/CUDA.")
            device = torch.device("cuda:0")
            device_text = f"GPU — {torch.cuda.get_device_name(0)}"
        else:
            device = torch.device("cpu")
            device_text = "CPU"

        app_cfg = _read_json(root / "configs" / "config.json")
        if threshold is not None:
            app_cfg["demo_threshold"] = float(threshold)

        started = time.perf_counter()
        result = predict_file(path, root, device=device, model_choice=model_choice, waveform=waveform)
        elapsed = time.perf_counter() - started
        probabilities = result["probabilities"][model_choice]
        rows = [{"Lớp": label, "Xác suất": float(probabilities[label])} for label in VI]
        probability_table = (
            pd.DataFrame(rows)
            .sort_values("Xác suất", ascending=False)
            .reset_index(drop=True)
        )
        top_index = CLASSES.index(result["top1_class"])
        confidence = float(probabilities[VI[top_index]])
        result["label"] = VI[top_index] if confidence >= float(app_cfg["demo_threshold"]) else "Ch\u01b0a x\u00e1c \u0111\u1ecbnh"
        headline = (
            "### Chưa chắc chắn — hãy thử âm thanh rõ hơn"
            if result["label"] == "Chưa xác định"
            else f"### {result['label']} • {confidence:.1%}"
        )

        model_text = MODEL_NAMES[model_choice]
        waveform_plot = _make_waveform_plot(waveform, sample_rate)
        total_elapsed = time.perf_counter() - request_started
        technical = (
            f"**Mô hình:** {model_text}  \n"
            f"**Thiết bị:** {device_text}  \n"
            f"**Thời lượng:** {duration:.2f} giây · **Cửa sổ:** {result['windows']} "
            "(4 giây, bước 2 giây)  \n"
            f"**Model + tiền xử lý:** {elapsed:.2f} giây · "
            f"**Tổng xử lý phía app:** {total_elapsed:.2f} giây  \n"
            f"**Tiền xử lý:** mono, 16 kHz; AST dùng ASTFeatureExtractor cục bộ đã lưu; "
            "xác suất các cửa sổ được lấy trung bình; model/checkpoint được giữ trong bộ nhớ "
            "cho các lần dự đoán tiếp theo. "
            f"Ngưỡng từ chối tham khảo: {app_cfg['demo_threshold']:.2f}. "
            "Ngưỡng này chưa được kiểm chứng như bộ phát hiện âm thanh ngoài 7 lớp."
        )
        status = (
            "Kết quả dự đoán xong."
            if result["label"] != "Chưa xác định"
            else "Điểm thấp hơn ngưỡng tham khảo; kết quả không khẳng định âm thanh ngoài 7 lớp."
        )
        return (
            headline,
            f"{confidence:.1%}",
            probability_table,
            technical,
            waveform_plot,
            status,
            waveform,
        )
    except (AudioInputError, FileNotFoundError, ValueError, RuntimeError) as exc:
        return ("### Chưa có kết quả", "—", None, "", None, f"⚠️ {exc}", None)


def clear_results():
    return (
        "### Chưa có kết quả",
        "**TÊN TỆP:** —",
        _empty_probabilities(),
        "",
        None,
        None,
        "Đã xóa kết quả.",
    )


def _probability_html(table, bars=False):
    rows = []
    for _, row in table.iterrows():
        probability = float(row["Xác suất"])
        bar = (f'<span class="bar-track"><span style="width:{probability * 100:.2f}%"></span></span>'
               if bars else "")
        rows.append(f'<tr><td>{html.escape(str(row["Lớp"]))}</td>'
                    f'<td><span>{probability:.2%}</span>{bar}</td></tr>')
    return ('<table class="prob-table"><thead><tr><th>Lớp</th><th>Xác suất</th></tr></thead>'
            '<tbody>' + "".join(rows) + '</tbody></table>')


def _empty_probabilities():
    return '<table class="prob-table"><tr><th>Lớp</th><th>Xác suất</th></tr>' + ''.join(
        f'<tr><td>{html.escape(label)}</td><td>\u2014</td></tr>' for label in VI
    ) + '</table>'


def _predict_dashboard(path, model, device, threshold, history):
    headline, confidence, probabilities, technical, waveform_plot, status, waveform = predict_audio(
        path, model, device, threshold
    )
    filename = f"**TÊN TỆP:** {html.escape(Path(path).name)}" if path else "**TÊN TỆP:** —"
    if probabilities is None:
        return (headline, filename, _empty_probabilities(), technical, waveform_plot, None,
                status, history, history)
    best = probabilities.iloc[0]
    label = str(best["Lớp"]) if float(best["Xác suất"]) >= threshold else "Chưa xác định"
    headline = f'### {label} • {confidence}'
    mel_plot = _make_logmel_plot(waveform, 16000)
    history = [[datetime.now().strftime("%H:%M:%S"), Path(path).name,
                f"{label} · {confidence}"]] + list(history or [])
    history = history[:50]
    return (headline, filename, _probability_html(probabilities), technical, waveform_plot,
            mel_plot, status, history, history)


def _toggle_logmel(visible: bool):
    next_visible = not visible
    return gr.update(visible=next_visible), next_visible, gr.update(value="◉")


def build_app():
    models = available_models()
    choices = [(MODEL_NAMES[kind], kind) for kind in models]
    default_model = "cnn4" if "cnn4" in models else (models[0] if models else None)
    if "cnn4" in models:
        warmup_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = _read_json(ROOT / "configs" / "config.json")
        warmup_prediction_model("cnn4", config, ROOT, config["test_folds"][0], warmup_device)
    app_cfg = _read_json(ROOT / "configs" / "app_config.json")
    reject_threshold = float(app_cfg.get("demo_threshold", 0.60))
    css = (ROOT / "src" / "dashboard.css").read_text(encoding="utf-8")
    theme = gr.themes.Base(primary_hue="indigo", neutral_hue="slate", font=["Arial", "sans-serif"]).set(
        body_background_fill="#0b111c", body_background_fill_dark="#0b111c",
        body_text_color="#e5e7eb", body_text_color_dark="#e5e7eb",
        block_background_fill="#15202e", block_background_fill_dark="#15202e",
        block_border_color="#303b49", block_border_color_dark="#303b49",
        input_background_fill="#253140", input_background_fill_dark="#253140",
        button_primary_background_fill="#403bd7", button_primary_background_fill_dark="#403bd7",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    )
    with gr.Blocks(title="Phân loại âm thanh đô thị", theme=theme, css=css) as app:
        with gr.Row(elem_classes=["page-header"]):
            gr.Markdown("# PHÂN LOẠI ÂM THANH ĐÔ THỊ", elem_id="page-title")
            theme_toggle = gr.Button(
                "☼",
                elem_id="theme-toggle",
                elem_classes=["icon-button"],
                size="sm",
                scale=0,
                min_width=44,
            )
            fullscreen_toggle = gr.Button(
                "⛶",
                elem_id="fullscreen-toggle",
                elem_classes=["icon-button"],
                size="sm",
                scale=0,
                min_width=44,
            )
        theme_toggle.click(
            fn=None,
            js="""() => {
                const light = !document.body.classList.contains("light-mode");
                document.body.classList.toggle("light-mode", light);
                document.documentElement.classList.toggle("light-mode", light);
                localStorage.setItem("urbanSoundTheme", light ? "Sáng" : "Tối");
                const button = document.querySelector("#theme-toggle");
                if (button) {
                    button.textContent = light ? "☾" : "☼";
                    button.setAttribute("aria-label", light ? "Chuyển sang giao diện tối" : "Chuyển sang giao diện sáng");
                    button.title = light ? "Chuyển sang giao diện tối" : "Chuyển sang giao diện sáng";
                }
                return [];
            }""",
        )
        fullscreen_toggle.click(
            fn=None,
            js="""() => {
                const toggle = document.querySelector("#fullscreen-toggle");
                const updateIcon = () => {
                    const active = Boolean(document.fullscreenElement);
                    if (toggle) {
                        toggle.setAttribute("aria-label", active ? "Thoát toàn màn hình" : "Bật toàn màn hình");
                        toggle.title = active ? "Thoát toàn màn hình" : "Bật toàn màn hình";
                    }
                };
                if (!document.documentElement.dataset.fullscreenButtonListener) {
                    document.addEventListener("fullscreenchange", updateIcon);
                    document.documentElement.dataset.fullscreenButtonListener = "true";
                }
                if (document.fullscreenElement) {
                    document.exitFullscreen().catch(error => console.error("Không thoát được chế độ toàn màn hình:", error));
                } else {
                    document.documentElement.requestFullscreen().catch(error => console.error("Không bật được chế độ toàn màn hình:", error));
                }
                return [];
            }""",
            show_progress="hidden",
        )
        app.load(
            fn=None,
            js="""() => {
                const light = localStorage.getItem("urbanSoundTheme") === "Sáng";
                document.body.classList.toggle("light-mode", light);
                document.documentElement.classList.toggle("light-mode", light);
                const button = document.querySelector("#theme-toggle");
                if (button) {
                    button.textContent = light ? "☾" : "☼";
                    button.setAttribute("aria-label", light ? "Chuyển sang giao diện tối" : "Chuyển sang giao diện sáng");
                    button.title = light ? "Chuyển sang giao diện tối" : "Chuyển sang giao diện sáng";
                }
                const fullscreen = document.querySelector("#fullscreen-toggle");
                if (fullscreen) {
                    fullscreen.setAttribute(
                        "aria-label",
                        document.fullscreenElement ? "Thoát toàn màn hình" : "Bật toàn màn hình",
                    );
                    fullscreen.title = fullscreen.getAttribute("aria-label");
                }
                const melToggle = document.querySelector("#mel-toggle");
                if (melToggle) {
                    melToggle.setAttribute("aria-label", "Hiện hoặc ẩn Log-Mel spectrogram");
                    melToggle.title = "Hiện hoặc ẩn Log-Mel spectrogram";
                }
                const upload = document.querySelector("#audio-player button[aria-label='Upload file']");
                const microphone = document.querySelector("#audio-player button[aria-label='Record audio']");
                const predict = document.querySelector("#predict-button");
                if (upload) {
                    upload.title = "Ctrl+O — Chọn tệp âm thanh";
                    upload.setAttribute("aria-keyshortcuts", "Control+O");
                }
                if (microphone) {
                    microphone.title = "Ctrl+M — Bật/tắt microphone";
                    microphone.setAttribute("aria-keyshortcuts", "Control+M");
                }
                if (predict) {
                    predict.title = "Ctrl+Enter — Dự đoán";
                    predict.setAttribute("aria-keyshortcuts", "Control+Enter");
                }
                if (!window.__urbanSoundShortcutsInstalled) {
                    document.addEventListener("keydown", event => {
                        if (!(event.ctrlKey || event.metaKey) || event.altKey || event.shiftKey) return;
                        const target = event.target;
                        if (target instanceof HTMLElement &&
                            (target.isContentEditable || target.matches("input, textarea, select, [role='textbox']"))) return;
                        const key = event.key.toLowerCase();
                        let control = null;
                        if (key === "o") {
                            control = document.querySelector("#audio-player button[aria-label='Upload file']");
                        } else if (key === "m") {
                            control = document.querySelector("#audio-player button[aria-label='Record audio'], #audio-player button[aria-label='Stop recording']");
                        } else if (event.key === "Enter") {
                            control = document.querySelector("#dashboard .actions button.primary");
                        }
                        if (control) {
                            event.preventDefault();
                            control.click();
                        }
                    }, true);
                    window.__urbanSoundShortcutsInstalled = true;
                }
                return [];
            }""",
            show_progress="hidden",
        )
        history_state = gr.State([])
        mel_visible = gr.State(False)
        with gr.Row(equal_height=False, elem_id="dashboard"):
            with gr.Column(scale=0, min_width=270, elem_classes=["history-sidebar"]):
                with gr.Column(elem_classes=["panel", "history-panel"]):
                    with gr.Row():
                        gr.Markdown("◷  Lịch sử dự đoán")
                        clear_history = gr.Button("⌫  Xóa", size="sm", scale=0, min_width=76)
                    history_table = gr.Dataframe(
                        headers=["Giờ", "Tệp", "Dự đoán"],
                        datatype=["str"] * 3,
                        value=[],
                        interactive=False,
                        wrap=True,
                        max_height=700,
                        column_widths=["20%", "40%", "40%"],
                    )
            with gr.Column(scale=3, min_width=340, elem_classes=["audio-main"]):
                audio = gr.Audio(label="Tệp âm thanh", sources=["upload", "microphone"],
                                 type="filepath", elem_id="audio-player")
                with gr.Row(elem_classes=["panel", "actions"]):
                    predict_button = gr.Button(
                        "▶  Dự đoán",
                        variant="primary",
                        elem_id="predict-button",
                    )
                    clear_button = gr.Button("↶  Xóa kết quả")
                status = gr.Markdown("Sẵn sàng. Chọn tệp âm thanh hoặc ghi âm để bắt đầu.", elem_id="status")
            with gr.Column(scale=3, min_width=340, elem_classes=["prediction-main"]):
                with gr.Column(elem_classes=["panel", "result-panel"]):
                    with gr.Row(elem_classes=["model-row"]):
                        headline = gr.Markdown("### Chưa có kết quả", elem_id="result-headline")
                        model = gr.Dropdown(choices=choices, value=default_model, label="Mô hình",
                                            interactive=bool(models), min_width=220, scale=1)
                    device = gr.Radio(
                        choices=DEVICE_CHOICES,
                        value="CUDA (GPU)" if torch.cuda.is_available() else "CPU",
                        label="Thiết bị",
                        elem_classes=["device-choice"],
                    )
                    filename = gr.Markdown("**TÊN TỆP:** —", elem_classes=["file-name"])
                    with gr.Accordion(f"ⓘ Ngưỡng từ chối: {reject_threshold:.2f} · Cài đặt dự đoán", open=False):
                        threshold = gr.Slider(0, 1, value=reject_threshold, step=0.01, label="Ngưỡng từ chối")
                    gr.Markdown("Xác suất — 7 lớp", elem_classes=["section-label"])
                    probabilities = gr.HTML(_empty_probabilities())
                with gr.Accordion("Thông tin kỹ thuật", open=False, elem_classes=["panel"]):
                    technical = gr.Markdown("Chưa có thông tin dự đoán.")
                    gr.Markdown(
                        "Phím tắt: **Ctrl+O** chọn tệp · **Ctrl+M** micro · **Ctrl+Enter** dự đoán",
                        elem_classes=["technical-shortcuts"],
                    )
                    waveform_plot = gr.Plot(label="Waveform", elem_classes=["audio-plot"])
                    mel_button = gr.Button(
                        "◉",
                        elem_id="mel-toggle",
                        elem_classes=["icon-button"],
                        size="sm",
                        min_width=44,
                    )
                    mel_plot = gr.Plot(label="Log-Mel spectrogram", elem_classes=["audio-plot"], visible=False)
                    mel_button.click(
                        _toggle_logmel,
                        inputs=mel_visible,
                        outputs=[mel_plot, mel_visible, mel_button],
                    )
        outputs = [headline, filename, probabilities, technical, waveform_plot, mel_plot, status,
                   history_state, history_table]
        predict_button.click(_predict_dashboard, [audio, model, device, threshold, history_state], outputs)
        clear_button.click(clear_results, outputs=outputs[:7])
        clear_history.click(lambda: ([], []), outputs=[history_state, history_table])
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", inbrowser=True)
