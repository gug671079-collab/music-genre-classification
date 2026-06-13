from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "length",
    "chroma_stft_mean",
    "chroma_stft_var",
    "rms_mean",
    "rms_var",
    "spectral_centroid_mean",
    "spectral_centroid_var",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_var",
    "rolloff_mean",
    "rolloff_var",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_var",
    "harmony_mean",
    "harmony_var",
    "perceptr_mean",
    "perceptr_var",
    "tempo",
    *[item for i in range(1, 21) for item in (f"mfcc{i}_mean", f"mfcc{i}_var")],
]


MEDIA_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def _require_librosa():
    try:
        import librosa
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: librosa. Install requirements.txt before extracting audio features."
        ) from exc
    return librosa


def _mean_var(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.var(values))


def _get_ffmpeg_executable() -> str | None:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
    except ModuleNotFoundError:
        return None

    return imageio_ffmpeg.get_ffmpeg_exe()


def convert_media_to_wav(input_path: Path, output_path: Path, sample_rate: int = 22050) -> Path:
    ffmpeg_exe = _get_ffmpeg_executable()
    if ffmpeg_exe is None:
        raise RuntimeError(
            "ffmpeg is required for MP4/video input. Install ffmpeg or add imageio-ffmpeg to requirements.txt."
        )

    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def load_audio_from_media(input_path: Path, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
    librosa = _require_librosa()
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix in MEDIA_EXTENSIONS:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "extracted_audio.wav"
            convert_media_to_wav(input_path, wav_path, sample_rate=sample_rate)
            audio, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
    elif suffix in AUDIO_EXTENSIONS:
        audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return audio, sr


def iter_segments(audio: np.ndarray, sr: int, segment_seconds: float = 3.0) -> Iterable[tuple[int, np.ndarray]]:
    segment_length = int(sr * segment_seconds)
    if segment_length <= 0:
        raise ValueError("segment_seconds must be positive")

    segment_count = len(audio) // segment_length
    for segment_id in range(segment_count):
        start = segment_id * segment_length
        end = start + segment_length
        yield segment_id, audio[start:end]


def extract_segment_features(segment: np.ndarray, sr: int) -> dict[str, float]:
    librosa = _require_librosa()

    chroma_stft = librosa.feature.chroma_stft(y=segment, sr=sr)
    rms = librosa.feature.rms(y=segment)
    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)
    harmony, perceptr = librosa.effects.hpss(segment)
    if hasattr(librosa.feature, "tempo"):
        tempo = librosa.feature.tempo(y=segment, sr=sr)
    else:
        tempo = librosa.beat.tempo(y=segment, sr=sr)
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)

    row: dict[str, float] = {"length": float(len(segment))}

    feature_map = {
        "chroma_stft": chroma_stft,
        "rms": rms,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "rolloff": rolloff,
        "zero_crossing_rate": zero_crossing_rate,
        "harmony": harmony,
        "perceptr": perceptr,
    }
    for name, values in feature_map.items():
        mean, var = _mean_var(values)
        row[f"{name}_mean"] = mean
        row[f"{name}_var"] = var

    row["tempo"] = float(np.ravel(tempo)[0])

    for index, values in enumerate(mfcc, start=1):
        mean, var = _mean_var(values)
        row[f"mfcc{index}_mean"] = mean
        row[f"mfcc{index}_var"] = var

    return row


def extract_features_from_media(
    input_path: Path,
    sample_rate: int = 22050,
    segment_seconds: float = 3.0,
    keep_partial: bool = False,
) -> pd.DataFrame:
    input_path = Path(input_path)
    audio, sr = load_audio_from_media(input_path, sample_rate=sample_rate)

    rows = []
    for segment_id, segment in iter_segments(audio, sr, segment_seconds=segment_seconds):
        row = extract_segment_features(segment, sr=sr)
        row["filename"] = f"{input_path.stem}.{segment_id}.wav"
        row["source_file"] = input_path.name
        row["segment_id"] = segment_id
        row["start_seconds"] = segment_id * segment_seconds
        row["end_seconds"] = (segment_id + 1) * segment_seconds
        rows.append(row)

    remainder_start = len(rows) * int(sr * segment_seconds)
    if keep_partial and remainder_start < len(audio):
        segment = audio[remainder_start:]
        if len(segment) > sr:
            row = extract_segment_features(segment, sr=sr)
            row["filename"] = f"{input_path.stem}.{len(rows)}.wav"
            row["source_file"] = input_path.name
            row["segment_id"] = len(rows)
            row["start_seconds"] = remainder_start / sr
            row["end_seconds"] = len(audio) / sr
            rows.append(row)

    df = pd.DataFrame(rows)
    metadata = ["filename", "source_file", "segment_id", "start_seconds", "end_seconds"]
    return df[metadata + FEATURE_COLUMNS] if not df.empty else pd.DataFrame(columns=metadata + FEATURE_COLUMNS)


def save_features(input_path: Path, output_csv: Path) -> Path:
    features = extract_features_from_media(input_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False, encoding="utf-8")
    return output_csv
