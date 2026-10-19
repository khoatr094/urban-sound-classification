from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from src import app_ui
from src.app_ui import AudioInputError, _make_waveform_plot, available_models, validate_audio
from src import engine
from src.engine import (
    CLASSES, audio_windows, available_checkpoint_kinds, predict_file,
    prediction_checkpoint_path,
)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> Path:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(pcm.tobytes())
    return path


def test_waveform_plot_covers_the_full_audio_duration():
    sample_rate = 16000
    duration_seconds = 45
    samples = np.zeros(sample_rate * duration_seconds, dtype=np.float32)

    figure = _make_waveform_plot(samples, sample_rate)

    try:
        axis = figure.axes[0]
        assert axis.get_xlim() == (0, duration_seconds)
        assert axis.get_title() == "Waveform"
        assert len(axis.lines[0].get_xdata()) <= 25000
        assert figure.get_size_inches()[1] >= 4
    finally:
        figure.clear()


def test_history_panel_and_fullscreen_control_are_available(monkeypatch):
    monkeypatch.setattr(app_ui, "available_models", lambda root=ROOT: [])

    config = app_ui.build_app().get_config_file()
    components = {component["id"]: component for component in config["components"]}
    history_table = next(
        component
        for component in components.values()
        if component["type"] == "dataframe"
        and component["props"].get("headers") == ["Giờ", "Tệp", "Dự đoán"]
    )
    fullscreen_button = next(
        component
        for component in components.values()
        if component["props"].get("elem_id") == "fullscreen-toggle"
    )
    technical_accordion = next(
        component
        for component in components.values()
        if component["type"] == "accordion"
        and component["props"].get("label") == "Thông tin kỹ thuật"
    )
    mel_button = next(
        component
        for component in components.values()
        if component["props"].get("elem_id") == "mel-toggle"
    )

    def find_layout_node(node, component_id):
        if node["id"] == component_id:
            return node
        for child in node.get("children", []):
            found = find_layout_node(child, component_id)
            if found is not None:
                return found
        return None

    def descendant_ids(node):
        return {node["id"]} | {
            child_id
            for child in node.get("children", [])
            for child_id in descendant_ids(child)
        }

    technical_ids = descendant_ids(find_layout_node(config["layout"], technical_accordion["id"]))
    chart_ids = {
        component["id"]
        for component in components.values()
        if component["type"] == "plot"
        and component["props"].get("label") in {"Waveform", "Log-Mel spectrogram"}
    }
    shortcuts = next(
        component
        for component in components.values()
        if "Phím tắt:" in str(component["props"].get("value", ""))
    )

    assert history_table["props"]["max_height"] == 700
    assert fullscreen_button["type"] == "button"
    assert chart_ids <= technical_ids
    assert mel_button["props"]["value"] == "◉"
    assert shortcuts["id"] in technical_ids
    assert any("requestFullscreen" in event.get("js", "") for event in config["dependencies"])


def test_audio_windows_cover_a_fifteen_minute_recording():
    sample_rate = 16000
    waveform = np.zeros(sample_rate * 15 * 60, dtype=np.uint8)

    windows = audio_windows(waveform, window_samples=sample_rate * 4, hop_samples=sample_rate * 2)

    assert len(windows) == 449
    assert all(len(window) == sample_rate * 4 for window in windows)


def test_predict_file_classifies_all_windows_of_fifteen_minute_audio(monkeypatch):
    sample_rate = 16000
    waveform = np.full(sample_rate * 15 * 60, 0.1, dtype=np.float32)
    processed_samples = []

    class StubModel:
        def __call__(self, batch):
            processed_samples.append(len(batch))
            return torch.zeros((len(batch), len(CLASSES)), device=batch.device)

    monkeypatch.setattr(engine, "_get_prediction_model", lambda *args, **kwargs: StubModel())
    monkeypatch.setattr(
        engine,
        "logmel",
        lambda *_: np.zeros((64, 251), dtype=np.float32),
    )

    result = predict_file(
        "fifteen-minutes.wav",
        ROOT,
        device=torch.device("cpu"),
        model_choice="cnn4",
        waveform=waveform,
    )

    assert result["windows"] == 449
    assert sum(processed_samples) == 449


@pytest.mark.parametrize(
    ("duration", "raises"),
    [(900.0, False), (900.1, True)],
)
def test_audio_duration_limit_is_fifteen_minutes(tmp_path, monkeypatch, duration, raises):
    path = tmp_path / "long-audio.wav"
    path.write_bytes(b"audio")
    monkeypatch.setattr(
        app_ui,
        "_read_json",
        lambda config_path: (
            {"max_duration_seconds": 900}
            if config_path.name == "app_config.json"
            else {"silence_rms": 0.0001}
        ),
    )
    monkeypatch.setattr(
        app_ui.sf,
        "info",
        lambda _: type("AudioInfo", (), {"frames": 1, "duration": duration})(),
    )
    monkeypatch.setattr(
        app_ui,
        "load_audio",
        lambda *args, **kwargs: np.full(16000, 0.1, dtype=np.float32),
    )

    if raises:
        with pytest.raises(AudioInputError, match="tối đa 900 giây"):
            validate_audio(path)
    else:
        waveform, sample_rate, actual_duration = validate_audio(path)
        assert sample_rate == 16000
        assert len(waveform) == 16000
        assert actual_duration == duration


def test_subsecond_audio_passes_audio_validation(tmp_path):
    sample_rate = 16000
    duration_seconds = 0.35
    samples = np.sin(
        2 * np.pi * 440 * np.arange(int(sample_rate * duration_seconds)) / sample_rate
    ).astype(np.float32) * 0.2
    path = _write_wav(tmp_path / "short-tone.wav", samples)

    waveform, actual_sample_rate, duration = validate_audio(path, ROOT)

    assert actual_sample_rate == sample_rate
    assert len(waveform) == len(samples)
    assert 0 < duration < 1


def test_flat_checkpoints_and_ensemble_are_available():
    assert prediction_checkpoint_path(ROOT, 10, "ast") == ROOT / "outputs/checkpoints/ast_best.pt"
    assert prediction_checkpoint_path(ROOT, 10, "cnn4") == ROOT / "outputs/checkpoints/cnn4_best.pt"
    assert prediction_checkpoint_path(ROOT, 10, "rescnn") == ROOT / "outputs/checkpoints/rescnn_best.pt"
    assert available_checkpoint_kinds(ROOT, 10) == ["cnn4", "rescnn", "ast"]
    assert available_models(ROOT) == ["cnn4", "rescnn", "ast", "ensemble"]
    assert (ROOT / "configs/ast_processor/preprocessor_config.json").is_file()


def test_prediction_model_is_loaded_once_per_checkpoint_and_device(monkeypatch):
    monkeypatch.setattr(engine, "_PREDICTION_MODELS", {})
    build_calls = []
    load_calls = []
    monkeypatch.setattr(
        engine,
        "build_model",
        lambda *args, **kwargs: build_calls.append((args, kwargs)) or torch.nn.Identity(),
    )
    monkeypatch.setattr(
        engine,
        "load_pt",
        lambda path: load_calls.append(path) or {"model": {}},
    )

    first = engine._get_prediction_model("cnn4", {}, ROOT, 10, torch.device("cpu"))
    second = engine._get_prediction_model("cnn4", {}, ROOT, 10, torch.device("cpu"))

    assert first is second
    assert len(build_calls) == len(load_calls) == 1


def test_valid_audio_is_decoded_as_mono_16khz(tmp_path):
    samples = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32) * 0.2
    path = _write_wav(tmp_path / "tone.wav", samples)

    waveform, sample_rate, duration = validate_audio(path, ROOT)

    assert sample_rate == 16000
    assert waveform.ndim == 1
    assert np.isfinite(waveform).all()
    assert math.isclose(duration, 1.0, abs_tol=0.01)


def test_silent_audio_is_rejected_with_a_clear_message(tmp_path):
    path = _write_wav(tmp_path / "silence.wav", np.zeros(16000, dtype=np.float32))

    with pytest.raises(AudioInputError, match="im lặng"):
        validate_audio(path, ROOT)


def test_empty_and_unsupported_files_are_rejected(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.touch()
    unsupported = tmp_path / "sound.aiff"
    unsupported.write_bytes(b"not audio")

    with pytest.raises(AudioInputError, match="rỗng"):
        validate_audio(empty, ROOT)
    with pytest.raises(AudioInputError, match="chưa hỗ trợ"):
        validate_audio(unsupported, ROOT)


def test_corrupt_audio_is_rejected_with_a_clear_message(tmp_path):
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"this is not a WAV file")

    with pytest.raises(AudioInputError, match="Không đọc được âm thanh"):
        validate_audio(corrupt, ROOT)


def test_all_four_models_predict_with_flat_checkpoints(tmp_path):
    sample_rate = 16000
    samples = np.sin(2 * np.pi * 800 * np.arange(sample_rate // 4) / sample_rate).astype(np.float32) * 0.2
    audio_path = _write_wav(tmp_path / "test.wav", samples)

    for model_choice in available_models(ROOT):
        result = predict_file(
            audio_path,
            ROOT,
            device=torch.device("cpu"),
            model_choice=model_choice,
        )
        values = list(result["probabilities"][model_choice].values())

        assert result["selected_model"] == model_choice
        assert result["windows"] == 1
        assert len(CLASSES) == len(values) == 7
        assert all(math.isfinite(value) for value in values)
        assert all(0.0 <= value <= 1.0 for value in values)
        assert math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5)
