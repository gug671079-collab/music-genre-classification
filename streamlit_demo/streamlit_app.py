from __future__ import annotations

import base64
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from audio_feature_extractor import extract_features_from_media, load_audio_from_media
from model_predictor import aggregate_genre_prediction, artifacts_available, predict_segments


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
REFERENCE_CSV = PROJECT_ROOT / "data" / "processed" / "processed_features_3_sec.csv"
ASSET_DIR = APP_DIR / "assets"
MONKEY_IMG = ASSET_DIR / "listen_meme_01.jpg"
ENJOY_IMG = ASSET_DIR / "listen_meme_enjoy.jpg"
CAT_IMG = ASSET_DIR / "listen_meme_cat.jpg"
LOW_IMG = ASSET_DIR / "listen_meme_low.jpg"
SAMPLE_AUDIO = APP_DIR / "local_samples" / "kaerou_30s.wav"

GENRE_NAMES = {
    "blues": "布鲁斯",
    "classical": "古典",
    "country": "乡村",
    "disco": "迪斯科",
    "hiphop": "嘻哈",
    "jazz": "爵士",
    "metal": "金属",
    "pop": "流行",
    "reggae": "雷鬼",
    "rock": "摇滚",
}


st.set_page_config(page_title="iListen 我听歌", page_icon="iL", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');
        :root {
            --acid: #dfff00;
            --ink: #f7f7f1;
            --muted: #9a9a8c;
            --panel: #151512;
            --line: rgba(223, 255, 0, 0.46);
        }
        html, body, [class*="stApp"] {
            background: #090909;
            color: var(--ink);
        }
        .block-container {
            max-width: 1220px;
            padding-top: 2.1rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3, p, span, label, div {
            font-family: "Microsoft YaHei", "Arial", sans-serif;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div,
        .stFileUploader section {
            background: #11110f !important;
            border: 1px solid var(--line) !important;
            color: var(--ink) !important;
        }
        .stButton button, .stDownloadButton button {
            background: var(--acid);
            color: #050505;
            border: 0;
            border-radius: 0;
            font-weight: 900;
        }
        .cover-stage-marker {
            display: none;
        }
        div[data-testid="stVerticalBlock"]:has(> div:first-child .cover-stage-marker) {
            border: 1px solid var(--line);
            background:
                linear-gradient(90deg, rgba(223,255,0,.12) 0 1px, transparent 1px 100%),
                repeating-linear-gradient(135deg, rgba(223,255,0,.06) 0 8px, transparent 8px 18px),
                linear-gradient(145deg, #11110f 0%, #090909 70%);
            background-size: 72px 72px, auto, auto;
            padding: clamp(28px, 4vw, 54px);
            min-height: min(760px, calc(100vh - 72px));
            position: relative;
            overflow: hidden;
            margin-bottom: 24px;
        }
        .hero-title {
            color: var(--ink);
            font-family: "Bebas Neue", "Arial Narrow", "Arial", sans-serif;
            font-size: clamp(118px, 19.5vw, 255px);
            font-weight: 400;
            line-height: .72;
            margin: 0;
            letter-spacing: .015em;
            position: relative;
            z-index: 1;
            white-space: nowrap;
            text-transform: none;
            text-shadow:
                5px 0 0 var(--acid),
                -1px -1px 0 #050505,
                1px 1px 0 #050505;
        }
        .hero-title:after {
            content: attr(data-shadow);
            position: absolute;
            left: 12px;
            top: 12px;
            color: transparent;
            -webkit-text-stroke: 1px rgba(223,255,0,.52);
            text-stroke: 1px rgba(223,255,0,.52);
            z-index: -1;
            white-space: nowrap;
        }
        .cover-upload {
            margin-top: clamp(100px, 19vh, 210px);
            padding: 0;
            position: relative;
            z-index: 2;
            max-width: 860px;
        }
        div[data-testid="stVerticalBlock"] > div:has(.cover-upload) + div[data-testid="stHorizontalBlock"],
        div[data-testid="stVerticalBlock"] > div:has(.cover-upload) + div.stHorizontalBlock {
            margin-top: 0;
            padding-left: 0;
            padding-right: 0;
            position: relative;
            z-index: 3;
        }
        div.stFileUploader {
            position: relative;
            z-index: 4;
        }
        div.stFileUploader section {
            min-height: 128px;
            position: relative;
        }
        div.stFileUploader section [data-testid="stFileUploaderDropzoneInstructions"] {
            visibility: hidden;
        }
        div.stFileUploader section:before {
            content: "把音乐文件拖到这里";
            position: absolute;
            left: 84px;
            top: 26px;
            color: var(--ink);
            font-size: 24px;
            font-weight: 900;
            pointer-events: none;
        }
        div.stFileUploader section:after {
            content: "单个文件不超过 200MB";
            position: absolute;
            left: 84px;
            top: 62px;
            color: var(--muted);
            font-size: 16px;
            pointer-events: none;
        }
        div.stFileUploader button {
            font-size: 0 !important;
        }
        div.stFileUploader button:after {
            content: "选择文件";
            font-size: 17px;
            font-weight: 900;
        }
        .cover-upload-title {
            color: var(--ink);
            font-size: clamp(26px, 3vw, 42px);
            font-weight: 900;
            line-height: 1.2;
            margin-bottom: 20px;
        }
        .format-line {
            color: #d8d8ce;
            font-size: 15px;
            line-height: 1.7;
            margin-top: 14px;
        }
        .format-line span {
            color: var(--acid);
            font-weight: 900;
        }
        .upload-under {
            margin-top: 12px;
            color: #d8d8ce;
            font-size: 15px;
            line-height: 1.7;
        }
        .upload-under-title {
            color: var(--ink);
            font-weight: 900;
            margin-bottom: 4px;
        }
        .upload-under span {
            color: var(--acid);
            font-weight: 900;
        }
        .panel {
            border: 1px solid var(--line);
            background: rgba(21, 21, 18, .92);
            padding: 18px;
            min-height: 100%;
        }
        .panel-title {
            color: var(--acid);
            font-weight: 900;
            font-size: 15px;
            margin-bottom: 10px;
        }
        .small-note {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.65;
        }
        .section-head {
            border-top: 1px solid var(--line);
            margin: 34px 0 18px;
            padding-top: 18px;
            display: block;
        }
        .section-title {
            color: var(--ink);
            font-family: "Microsoft YaHei", "Arial", sans-serif;
            font-size: clamp(42px, 5.8vw, 76px);
            font-weight: 900;
            line-height: .98;
            letter-spacing: 0;
            text-shadow: 3px 0 0 rgba(223,255,0,.85);
        }
        .source-title {
            font-family: "Bebas Neue", "Arial Narrow", "Arial", sans-serif;
            font-size: clamp(86px, 11vw, 150px);
            font-weight: 400;
            line-height: .8;
            text-transform: lowercase;
            text-shadow: 5px 0 0 rgba(223,255,0,.85);
        }
        .source-lines {
            color: #d8d8ce;
            font-size: 18px;
            line-height: 2;
            font-weight: 800;
        }
        .source-lines span {
            color: var(--acid);
            font-weight: 900;
        }
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin: 12px 0 18px;
        }
        .analysis-tile {
            border: 1px solid rgba(223,255,0,.38);
            background:
                linear-gradient(90deg, rgba(223,255,0,.08) 0 1px, transparent 1px 100%),
                repeating-linear-gradient(135deg, rgba(223,255,0,.045) 0 7px, transparent 7px 17px),
                #11110f;
            background-size: 46px 46px, auto, auto;
            padding: 14px 16px;
            min-height: 96px;
        }
        .analysis-label {
            color: var(--acid);
            font-size: 13px;
            font-weight: 900;
            margin-bottom: 12px;
        }
        .analysis-value {
            color: var(--ink);
            font-size: clamp(30px, 4vw, 48px);
            font-weight: 900;
            line-height: .9;
        }
        .analysis-unit {
            color: #d8d8ce;
            font-size: 14px;
            font-weight: 800;
            margin-left: 6px;
        }
        .chart-title {
            border: 1px solid var(--line);
            border-bottom: 0;
            background: #11110f;
            color: var(--acid);
            display: inline-block;
            padding: 8px 12px;
            font-size: 13px;
            font-weight: 900;
            margin-top: 4px;
        }
        .radar-title {
            border: 1px solid var(--line);
            background: #11110f;
            color: var(--acid);
            display: inline-block;
            padding: 12px 18px;
            font-size: clamp(20px, 2.4vw, 34px);
            font-weight: 900;
            margin: 8px 0 10px;
            text-shadow: 2px 0 0 rgba(255,255,255,.18);
        }
        .genre-neighbor-panel {
            border: 1px solid var(--line);
            background:
                linear-gradient(90deg, rgba(223,255,0,.09) 0 1px, transparent 1px 100%),
                repeating-linear-gradient(135deg, rgba(223,255,0,.055) 0 7px, transparent 7px 17px),
                #11110f;
            background-size: 48px 48px, auto, auto;
            padding: 16px;
            margin-top: 14px;
        }
        .genre-neighbor-title {
            color: var(--acid);
            font-size: 16px;
            font-weight: 900;
            margin-bottom: 4px;
        }
        .genre-neighbor-note {
            color: #9a9a8c;
            font-size: 12px;
            font-weight: 800;
            margin-bottom: 12px;
        }
        .genre-row {
            display: grid;
            grid-template-columns: 42px 86px minmax(0, 1fr) 52px;
            gap: 12px;
            align-items: center;
            border-top: 1px solid rgba(223,255,0,.22);
            padding: 12px 0;
        }
        .genre-rank {
            color: var(--acid);
            font-size: 24px;
            font-weight: 900;
            line-height: 1;
        }
        .genre-name {
            color: var(--ink);
            font-size: 18px;
            font-weight: 900;
        }
        .genre-score {
            color: #d8d8ce;
            font-size: 14px;
            font-weight: 900;
            text-align: right;
        }
        .genre-track {
            height: 14px;
            border: 1px solid rgba(223,255,0,.42);
            background: #1b1d13;
            box-shadow: inset 0 0 14px rgba(0,0,0,.55);
        }
        .genre-fill {
            height: 100%;
            background:
                repeating-linear-gradient(135deg, rgba(5,5,5,.22) 0 4px, transparent 4px 9px),
                linear-gradient(90deg, rgba(223,255,0,.36), var(--acid));
            box-shadow: 0 0 20px rgba(223,255,0,.50);
        }
        .score-wrap {
            border: 1px solid var(--line);
            background:
                linear-gradient(90deg, rgba(223,255,0,.09) 0 1px, transparent 1px 100%),
                repeating-linear-gradient(135deg, rgba(223,255,0,.055) 0 8px, transparent 8px 18px),
                #11110f;
            background-size: 52px 52px, auto, auto;
            padding: 28px;
            min-height: 292px;
            position: relative;
            overflow: hidden;
        }
        .score-label {
            color: var(--acid);
            font-size: 16px;
            font-weight: 900;
        }
        .score-number {
            color: var(--ink);
            font-size: clamp(84px, 13vw, 160px);
            font-weight: 900;
            line-height: .86;
        }
        .score-suffix {
            color: var(--acid);
            font-size: 34px;
            font-weight: 900;
        }
        .score-genre-bars {
            border-top: 1px solid rgba(223,255,0,.26);
            margin-top: 46px;
            padding-top: 14px;
        }
        .score-bars-title {
            color: var(--acid);
            font-size: 13px;
            font-weight: 900;
            margin-bottom: 12px;
        }
        .score-bars-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            align-items: end;
            min-height: 150px;
        }
        .score-bar-item {
            min-width: 0;
            display: grid;
            grid-template-rows: 24px 96px auto;
            gap: 7px;
            align-items: end;
        }
        .score-bar-value {
            color: #d8d8ce;
            font-size: 13px;
            font-weight: 900;
            line-height: 1;
        }
        .score-bar-shell {
            height: 96px;
            border: 1px solid rgba(223,255,0,.36);
            background:
                linear-gradient(90deg, rgba(223,255,0,.10) 0 1px, transparent 1px 100%),
                #171912;
            background-size: 16px 16px;
            display: flex;
            align-items: flex-end;
            box-shadow: inset 0 0 18px rgba(0,0,0,.60);
        }
        .score-bar-fill {
            width: 100%;
            min-height: 8px;
            background:
                repeating-linear-gradient(135deg, rgba(5,5,5,.22) 0 4px, transparent 4px 9px),
                linear-gradient(180deg, var(--acid), rgba(223,255,0,.30));
            box-shadow: 0 0 18px rgba(223,255,0,.55);
        }
        .score-bar-item:not(:first-child) .score-bar-fill {
            opacity: .68;
        }
        .score-bar-name {
            color: var(--ink);
            font-size: 13px;
            font-weight: 900;
            line-height: 1.15;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .axis-panel {
            border: 1px solid var(--line);
            background:
                linear-gradient(90deg, rgba(223,255,0,.07) 0 1px, transparent 1px 100%),
                #11110f;
            background-size: 52px 52px;
            padding: 22px;
        }
        .axis-inline-title {
            color: var(--acid);
            font-size: 18px;
            font-weight: 900;
            margin: 0 0 2px;
            line-height: 1;
        }
        .metric-row {
            border: 1px solid rgba(223,255,0,.42);
            background:
                linear-gradient(90deg, rgba(223,255,0,.09) 0 1px, transparent 1px 100%),
                repeating-linear-gradient(135deg, rgba(223,255,0,.06) 0 7px, transparent 7px 17px),
                #11110f;
            background-size: 44px 44px, auto, auto;
            padding: 16px;
            margin: 6px 0 14px;
            position: relative;
            overflow: hidden;
        }
        .metric-head {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 18px;
            color: var(--ink);
            font-weight: 900;
        }
        .metric-side:last-child {
            text-align: right;
        }
        .metric-name {
            color: var(--ink);
            font-size: 18px;
            font-weight: 900;
            line-height: 1.2;
        }
        .metric-desc {
            color: #9a9a8c;
            font-size: 12px;
            font-weight: 700;
            line-height: 1.45;
            margin-top: 5px;
        }
        .metric-score {
            color: var(--acid);
            font-size: 42px;
            font-weight: 900;
            line-height: .9;
            margin: 14px 0 10px;
            text-shadow: 3px 0 0 rgba(255,255,255,.18);
        }
        .metric-score span {
            color: #d8d8ce;
            font-size: 13px;
            font-weight: 900;
            margin-left: 5px;
        }
        .bar-track {
            height: 16px;
            margin-top: 8px;
            background:
                linear-gradient(90deg, rgba(223,255,0,.16) 0 1px, transparent 1px 100%),
                #1f2115;
            background-size: 18px 18px;
            border: 1px solid rgba(223,255,0,.48);
            position: relative;
            box-shadow: inset 0 0 18px rgba(0,0,0,.55);
        }
        .bar-fill {
            height: 100%;
            background:
                repeating-linear-gradient(135deg, rgba(5,5,5,.22) 0 4px, transparent 4px 9px),
                linear-gradient(90deg, rgba(223,255,0,.38), var(--acid));
            box-shadow: 0 0 24px rgba(223,255,0,.55);
        }
        .bar-pin {
            position: absolute;
            top: -10px;
            width: 5px;
            height: 34px;
            background: #fff;
            box-shadow: 0 0 18px rgba(223,255,0,.95), 0 0 34px rgba(223,255,0,.55);
        }
        @media (max-width: 720px) {
            .metric-head {
                gap: 10px;
            }
            .metric-name {
                font-size: 16px;
            }
            .metric-desc {
                font-size: 11px;
            }
            .metric-score {
                font-size: 36px;
            }
        }
        .verdict {
            border: 1px solid var(--line);
            border-left: 8px solid var(--acid);
            background:
                repeating-linear-gradient(135deg, rgba(223,255,0,.055) 0 7px, transparent 7px 18px),
                rgba(223,255,0,.06);
            padding: 20px 22px;
            margin-top: 16px;
            font-size: 20px;
            line-height: 1.7;
            font-weight: 900;
        }
        .loading-box {
            border: 1px solid var(--line);
            background:
                linear-gradient(90deg, rgba(223,255,0,.10) 0 1px, transparent 1px 100%),
                repeating-linear-gradient(135deg, rgba(223,255,0,.07) 0 8px, transparent 8px 18px),
                #11110f;
            background-size: 54px 54px, auto, auto;
            padding: 28px;
            display: grid;
            grid-template-columns: 230px 1fr;
            gap: 30px;
            align-items: center;
            margin: 28px 0;
            position: relative;
            overflow: hidden;
        }
        .loading-box:after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, transparent, rgba(223,255,0,.16), transparent);
            transform: translateX(-100%);
            animation: scanLine 1.8s linear infinite;
            pointer-events: none;
        }
        .loading-monkey {
            width: 220px;
            height: 220px;
            object-fit: cover;
            border: 1px solid var(--line);
            animation: spinMonkey 2.1s linear infinite;
        }
        @keyframes spinMonkey {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        @keyframes scanLine {
            to { transform: translateX(100%); }
        }
        .loading-title {
            color: var(--acid);
            font-size: clamp(36px, 5vw, 70px);
            line-height: 1.05;
            font-weight: 900;
            margin-bottom: 14px;
        }
        .loading-steps {
            color: var(--ink);
            font-size: 18px;
            line-height: 1.9;
            font-weight: 800;
        }
        .report-card {
            border: 1px solid var(--line);
            background:
                linear-gradient(90deg, rgba(223,255,0,.08) 0 1px, transparent 1px 100%),
                linear-gradient(145deg, #151512 0%, #090909 68%),
                #11110f;
            background-size: 58px 58px, auto, auto;
            padding: 24px;
            margin-top: 26px;
            position: relative;
            overflow: hidden;
        }
        .report-card:before {
            content: "iListen";
            position: absolute;
            right: 18px;
            top: 10px;
            color: rgba(223,255,0,.12);
            font-size: 92px;
            font-weight: 900;
            line-height: 1;
        }
        .card-title {
            color: var(--acid);
            font-size: 15px;
            font-weight: 900;
            margin-bottom: 8px;
        }
        .goat-banner {
            color: var(--ink);
            font-size: clamp(28px, 4.2vw, 54px);
            font-weight: 900;
            line-height: 1.05;
            margin: 6px 0 14px;
            text-shadow:
                4px 0 0 var(--acid),
                0 0 18px rgba(223,255,0,.42);
        }
        .card-score {
            color: var(--ink);
            font-size: 86px;
            font-weight: 900;
            line-height: .9;
            margin: 10px 0 12px;
        }
        .card-meta {
            color: #d8d8ce;
            font-size: 16px;
            font-weight: 800;
            margin-bottom: 18px;
        }
        .card-verdict {
            color: var(--ink);
            font-size: 18px;
            line-height: 1.65;
            font-weight: 800;
            max-width: 820px;
        }
        .card-body {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 18px;
            align-items: end;
        }
        .card-body.has-meme {
            grid-template-columns: minmax(0, 1fr) 170px;
        }
        .card-meme {
            width: 170px;
            height: 170px;
            object-fit: cover;
            border: 1px solid var(--line);
            filter: saturate(1.08) contrast(1.05);
            justify-self: end;
        }
        .card-axis {
            color: #d8d8ce;
            font-size: 14px;
            line-height: 1.9;
            margin-top: 12px;
        }
        @media (max-width: 720px) {
            .card-body.has-meme {
                grid-template-columns: 1fr;
            }
            .card-meme {
                width: 140px;
                height: 140px;
                justify-self: start;
            }
        }
        .footer-note {
            color: #7f7f73;
            font-size: 12px;
            border-top: 1px solid rgba(223,255,0,.25);
            padding-top: 16px;
            margin-top: 26px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_reference_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def image_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower().replace(".", "")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    return f"data:image/{mime};base64,{encoded}"


def feature_percentile(value: float, reference: pd.Series) -> float:
    clean = reference.dropna().astype(float)
    if clean.empty:
        return 50.0
    percentile = (clean <= value).mean() * 100
    return float(np.clip(percentile, 0, 100))


def mean_percentile(features: pd.DataFrame, reference: pd.DataFrame, columns: list[str]) -> float:
    scores = []
    for column in columns:
        if column in features.columns and column in reference.columns:
            scores.append(feature_percentile(float(features[column].mean()), reference[column]))
    return float(np.mean(scores)) if scores else 50.0


def load_audio_for_plot(media_path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    audio, sr = load_audio_from_media(media_path, sample_rate=sample_rate)
    if len(audio) > 100_000:
        step = int(np.ceil(len(audio) / 100_000))
        audio = audio[::step]
    return audio, sr


def plot_waveform(media_path: Path, sample_rate: int) -> go.Figure:
    audio, sr = load_audio_for_plot(media_path, sample_rate)
    seconds = np.arange(len(audio)) / sr
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=seconds,
            y=audio,
            mode="lines",
            line=dict(color="#dfff00", width=1),
            fill="tozeroy",
            fillcolor="rgba(223,255,0,.15)",
        )
    )
    fig.update_layout(
        title="",
        paper_bgcolor="#090909",
        plot_bgcolor="#090909",
        font=dict(color="#f7f7f1", family="Microsoft YaHei"),
        xaxis=dict(title="时间 / 秒", gridcolor="rgba(223,255,0,.12)"),
        yaxis=dict(title="振幅", gridcolor="rgba(223,255,0,.12)"),
        height=300,
        margin=dict(l=24, r=24, t=24, b=28),
    )
    return fig


def plot_top3(genre_scores: pd.DataFrame) -> go.Figure:
    top3 = genre_scores.head(3).copy()
    top3["display"] = top3["genre"].map(lambda item: GENRE_NAMES.get(item, item))
    top3 = top3.sort_values("score", ascending=True)
    fig = px.bar(
        top3,
        x="score",
        y="display",
        orientation="h",
        text=top3["score"].map(lambda value: f"{value:.2f}"),
        labels={"score": "平均概率", "display": ""},
    )
    fig.update_traces(marker_color="#dfff00", textposition="outside", cliponaxis=False)
    fig.update_layout(
        title="",
        paper_bgcolor="#090909",
        plot_bgcolor="#090909",
        font=dict(color="#f7f7f1", family="Microsoft YaHei"),
        xaxis=dict(range=[0, max(1.0, float(top3["score"].max()) * 1.18)], gridcolor="rgba(223,255,0,.12)"),
        yaxis=dict(gridcolor="rgba(223,255,0,.08)"),
        height=300,
        margin=dict(l=20, r=54, t=24, b=28),
    )
    return fig


def plot_genre_radar(genre_scores: pd.DataFrame) -> go.Figure:
    score_map = {str(row.genre): float(row.score) for row in genre_scores.itertuples(index=False)}
    keys = list(GENRE_NAMES.keys())
    labels = [GENRE_NAMES[key] for key in keys]
    values = [score_map.get(key, 0.0) for key in keys]
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]
    radial_max = max(0.35, max(values_closed) * 1.22)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            mode="lines+markers",
            fill="toself",
            fillcolor="rgba(223,255,0,.18)",
            line=dict(color="#dfff00", width=3),
            marker=dict(color="#f7f7f1", size=7, line=dict(color="#dfff00", width=2)),
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="",
        paper_bgcolor="#090909",
        plot_bgcolor="#090909",
        font=dict(color="#f7f7f1", family="Microsoft YaHei", size=16),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0, radial_max],
                showline=False,
                gridcolor="rgba(223,255,0,.20)",
                tickfont=dict(color="rgba(247,247,241,.62)", size=11),
            ),
            angularaxis=dict(
                gridcolor="rgba(223,255,0,.18)",
                linecolor="rgba(223,255,0,.35)",
                tickfont=dict(color="#f7f7f1", size=16),
            ),
        ),
        height=540,
        margin=dict(l=52, r=52, t=38, b=38),
        showlegend=False,
    )
    return fig


def render_genre_neighbors(genre_scores: pd.DataFrame) -> None:
    top3 = genre_scores.head(3).copy()
    rows = []
    for index, row in enumerate(top3.itertuples(index=False), start=1):
        genre = GENRE_NAMES.get(str(row.genre), str(row.genre))
        score = float(row.score)
        width = np.clip(score * 100, 0, 100)
        rows.append(
            f'<div class="genre-row"><div class="genre-rank">{index:02d}</div>'
            f'<div class="genre-name">{genre}</div><div class="genre-track">'
            f'<div class="genre-fill" style="width:{width:.1f}%"></div></div>'
            f'<div class="genre-score">{score:.2f}</div></div>'
        )
    st.markdown(
        '<div class="genre-neighbor-panel">'
        '<div class="genre-neighbor-title">GTZAN 十类风格近邻</div>'
        '<div class="genre-neighbor-note">先判风格，再开锐评。</div>'
        f'{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def render_score_genre_bars(genre_scores: pd.DataFrame) -> str:
    top3 = genre_scores.head(3).copy()
    max_score = max(float(top3["score"].max()), 0.001) if not top3.empty else 1.0
    items = []
    for row in top3.itertuples(index=False):
        genre = GENRE_NAMES.get(str(row.genre), str(row.genre))
        score = float(row.score)
        height = np.clip(score / max_score * 100, 12, 100)
        items.append(
            f'<div class="score-bar-item"><div class="score-bar-value">{score:.2f}</div>'
            f'<div class="score-bar-shell"><div class="score-bar-fill" style="height:{height:.1f}%"></div></div>'
            f'<div class="score-bar-name">{genre}</div></div>'
        )
    return (
        '<div class="score-genre-bars">'
        '<div class="score-bars-title">风格近邻</div>'
        f'<div class="score-bars-grid">{"".join(items)}</div>'
        '</div>'
    )


def style_scores(features: pd.DataFrame, reference: pd.DataFrame, genre_scores: pd.DataFrame) -> dict[str, float]:
    top = float(genre_scores.iloc[0]["score"]) if not genre_scores.empty else 0.35
    second = float(genre_scores.iloc[1]["score"]) if len(genre_scores) > 1 else 0.0
    gap = max(top - second, 0)
    style_clear = np.clip(35 + top * 45 + gap * 45, 0, 100)

    energy = mean_percentile(
        features,
        reference,
        ["rms_mean", "tempo", "zero_crossing_rate_mean"],
    )
    outward = mean_percentile(
        features,
        reference,
        ["spectral_centroid_mean", "rolloff_mean", "chroma_stft_mean"],
    )

    raw_total = np.clip(style_clear * 0.30 + energy * 0.30 + outward * 0.40, 0, 100)
    friendly_total = np.clip(65 + raw_total * 0.31, 65, 96)
    return {
        "style_clear": float(style_clear),
        "energy": float(energy),
        "outward": float(outward),
        "raw_total": float(raw_total),
        "total": float(friendly_total),
    }


def apply_curated_sample_score(scores: dict[str, float]) -> dict[str, float]:
    adjusted = scores.copy()
    adjusted["total"] = max(float(adjusted["total"]), 93.0)
    return adjusted


def curated_sample_verdict() -> str:
    return (
        "这才是真正的音乐！帰ろう不是乡村，别让模型带跑。"
        "副歌一到，弦乐和人声一起抬起来，像人从地上被风托住。"
        "有点伤感，但不是往下坠，是乘风归去。"
    )


def metric_bar(left: str, right: str, score: float) -> None:
    position = float(np.clip(score, 0, 100))
    descriptions = {
        "集百家长": "数值越低，代表风格越多元杂糅。",
        "风格鲜明": "数值越高，代表模型判断越集中。",
        "松弛耐听": "数值越低，代表听感更放松耐听。",
        "热烈直给": "数值越高，代表节奏和能量更直接。",
        "低调内敛": "数值越低，代表音色更收、更柔和。",
        "锋芒外放": "数值越高，代表声音更亮、更有存在感。",
    }
    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-head">
            <div class="metric-side">
              <div class="metric-name">{left}</div>
              <div class="metric-desc">{descriptions.get(left, "")}</div>
            </div>
            <div class="metric-side">
              <div class="metric-name">{right}</div>
              <div class="metric-desc">{descriptions.get(right, "")}</div>
            </div>
          </div>
          <div class="metric-score">{position:.0f}<span>/ 100</span></div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{position:.1f}%"></div>
            <div class="bar-pin" style="left:calc({position:.1f}% - 1px)"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def axis_summary(scores: dict[str, float]) -> str:
    style = "风格鲜明" if scores["style_clear"] >= 55 else "集百家长"
    energy = "热烈直给" if scores["energy"] >= 55 else "松弛耐听"
    outward = "锋芒外放" if scores["outward"] >= 55 else "低调内敛"
    return f"{style} / {energy} / {outward}"


def verdict_text(scores: dict[str, float], predicted_genre: str, track_name: str) -> str:
    title = track_name.strip() or "这首歌"
    genre = GENRE_NAMES.get(predicted_genre, predicted_genre)
    total = scores["total"]

    if scores["style_clear"] >= 72:
        style_part = f"路子很清楚，iListen 听着像{genre}。"
    elif scores["style_clear"] <= 42:
        style_part = "它不走单一路数，混得还挺顺。"
    else:
        style_part = f"有点{genre}味，但没把自己框死。"

    if scores["energy"] >= 70:
        energy_part = "劲儿给得挺满，适合把音量加两格。"
    elif scores["energy"] <= 38:
        energy_part = "不急，慢慢听反而对。"
    else:
        energy_part = "不炸，也不虚，刚好。"

    if scores["outward"] >= 70:
        color_part = "声音挺亮，存在感不低。"
    elif scores["outward"] <= 35:
        color_part = "锋芒收着，耐听型。"
    else:
        color_part = "表情不夸张，但有记忆点。"

    if total >= 85:
        ending = "这首可以，别划走。"
    elif total >= 76:
        ending = "有点东西，可以多听一遍。"
    elif total >= 66:
        ending = "不算爆，但不是白听。"
    else:
        ending = "先别急着跳过，可能是慢热。"

    return f"{title}：{style_part}{energy_part}{color_part}{ending}"


def render_hero() -> None:
    st.markdown(
        """
        <div class="cover-stage-marker"></div>
        <div class="hero-title" data-shadow="i Listen">i Listen</div>
        """,
        unsafe_allow_html=True,
    )


def render_loading() -> None:
    monkey_uri = image_to_data_uri(MONKEY_IMG)
    image_html = f'<img class="loading-monkey" src="{monkey_uri}" />' if monkey_uri else ""
    st.markdown(
        f"""
        <div class="loading-box">
          <div>{image_html}</div>
          <div>
            <div class="loading-title">I listen</div>
            <div class="loading-steps">
              01 正在拆音轨<br>
              02 正在听前奏<br>
              03 正在和数据集对暗号<br>
              04 正在组织锐评
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str) -> None:
    st.markdown(
        f"""
        <div class="section-head">
          <div class="section-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_snapshot(features: pd.DataFrame) -> None:
    tempo = float(features["tempo"].mean())
    rms = float(features["rms_mean"].mean())
    segment_count = len(features)
    st.markdown(
        f"""
        <div class="analysis-grid">
          <div class="analysis-tile">
            <div class="analysis-label">BPM</div>
            <div><span class="analysis-value">{tempo:.1f}</span></div>
          </div>
          <div class="analysis-tile">
            <div class="analysis-label">LOUDNESS</div>
            <div><span class="analysis-value">{rms:.4f}</span></div>
          </div>
          <div class="analysis-tile">
            <div class="analysis-label">有效采样切片</div>
            <div><span class="analysis-value">{segment_count}</span><span class="analysis-unit">段</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_upload_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def card_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def wrap_text_for_image(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        trial = current + char
        width = draw.textbbox((0, 0), trial, font=font)[2]
        if width <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def paste_square_image(canvas: Image.Image, image_path: Path, box: tuple[int, int, int]) -> None:
    if not image_path.exists():
        return
    x, y, size = box
    image = Image.open(image_path).convert("RGB")
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((size, size))
    canvas.paste(image, (x, y))


def report_card_png(
    track_name: str,
    artist_name: str,
    uploaded_name: str,
    score: float,
    predicted_display: str,
    scores: dict[str, float],
    verdict: str,
) -> bytes:
    width, height = 1200, 760
    acid = (223, 255, 0)
    ink = (247, 247, 241)
    muted = (172, 172, 156)
    line = (116, 134, 0)
    bg = (8, 8, 8)

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    for x in range(0, width, 58):
        draw.line((x, 0, x, height), fill=(24, 30, 3), width=1)
    for offset in range(-height, width, 22):
        draw.line((offset, height, offset + height, 0), fill=(19, 24, 2), width=9)
    draw.rectangle((24, 24, width - 24, height - 24), outline=line, width=2)

    title_font = card_font(28, True)
    goat_font = card_font(58, True)
    meta_font = card_font(28, True)
    score_font = card_font(116, True)
    suffix_font = card_font(38, True)
    axis_font = card_font(24, True)
    verdict_font = card_font(28, True)

    y = 58
    draw.text((62, y), "iListen 锐评卡", fill=acid, font=title_font)
    y += 46
    if score >= 85:
        draw.text((62 + 4, y), "这才是真正的音乐！", fill=acid, font=goat_font)
        draw.text((62, y), "这才是真正的音乐！", fill=ink, font=goat_font)
        y += 72

    title = track_name.strip() or Path(uploaded_name).stem
    artist = artist_name.strip()
    meta = f"{artist} - {title}" if artist else title
    draw.text((62, y), meta[:42], fill=muted, font=meta_font)
    y += 54

    score_text = f"{score:.0f}"
    draw.text((62, y), score_text, fill=ink, font=score_font)
    score_width = draw.textbbox((0, 0), score_text, font=score_font)[2]
    draw.text((72 + score_width, y + 58), "分", fill=acid, font=suffix_font)
    y += 132

    draw.text((62, y), f"风格分类：{predicted_display}", fill=muted, font=axis_font)
    y += 42
    axis_text = (
        f"集百家长 ↔ 风格鲜明：{scores['style_clear']:.0f}    "
        f"松弛耐听 ↔ 热烈直给：{scores['energy']:.0f}    "
        f"低调内敛 ↔ 锋芒外放：{scores['outward']:.0f}"
    )
    draw.text((62, y), axis_text, fill=muted, font=axis_font)
    y += 56

    meme_path = ENJOY_IMG if score >= 85 else CAT_IMG if score >= 75 else LOW_IMG
    paste_square_image(image, meme_path, (940, 466, 190))
    draw.rectangle((940, 466, 1130, 656), outline=line, width=2)

    for line_text in wrap_text_for_image(draw, verdict, verdict_font, 820)[:5]:
        draw.text((62, y), line_text, fill=ink, font=verdict_font)
        y += 42

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def render_report_card(
    track_name: str,
    artist_name: str,
    uploaded_name: str,
    score: float,
    predicted_display: str,
    scores: dict[str, float],
    verdict: str,
) -> None:
    title = track_name.strip() or Path(uploaded_name).stem
    artist = artist_name.strip()
    meta = f"{artist} - {title}" if artist else title
    meme_path = ENJOY_IMG if score >= 85 else CAT_IMG if score >= 75 else LOW_IMG
    meme_uri = image_to_data_uri(meme_path) if meme_path else ""
    meme_html = f'<img class="card-meme" src="{meme_uri}" />' if meme_uri else ""
    body_class = "card-body has-meme" if meme_html else "card-body"
    goat_html = '<div class="goat-banner">这才是真正的音乐！</div>' if score >= 85 else ""
    st.markdown(
        f"""
        <div class="report-card">
          <div class="card-title">iListen 锐评卡</div>
          {goat_html}
          <div class="card-meta">{meta}</div>
          <div class="card-score">{score:.0f} 分</div>
          <div class="card-meta">风格分类：{predicted_display}</div>
          <div class="card-axis">
            集百家长 ↔ 风格鲜明：{scores["style_clear"]:.0f}<br>
            松弛耐听 ↔ 热烈直给：{scores["energy"]:.0f}<br>
            低调内敛 ↔ 锋芒外放：{scores["outward"]:.0f}
          </div>
          <div class="{body_class}">
            <div class="card-verdict">{verdict}</div>
            {meme_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()

    with st.container():
        render_hero()

        st.markdown(
            """
            <div class="cover-upload">
              <div class="cover-upload-title">你上传，我聆听，我锐评。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        form_left, form_right = st.columns([1.25, 0.75], gap="large")
        with form_left:
            uploaded_file = st.file_uploader(
                "上传 MP4 或音频文件",
                type=["mp4", "mov", "m4v", "wav", "mp3", "flac", "ogg", "m4a"],
                help="V1 不自动抓歌，先用上传文件保证稳定。",
                label_visibility="collapsed",
            )
            st.markdown(
                """
                <div class="upload-under">
                  <div class="upload-under-title">上传 MP4 或音频文件</div>
                  <div>也支持 <span>MP4</span> / <span>MOV</span> 这类视频，以及 <span>MP3</span> / <span>WAV</span> / <span>FLAC</span> / <span>M4A</span> 等常见音频。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            track_name = ""
            artist_name = ""
        with form_right:
            st.markdown(
                """
                <div class="panel">
                  <div class="panel-title">样例</div>
                  <div class="small-note">
                    提供试听加锐评。
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            use_sample = False
            if SAMPLE_AUDIO.exists():
                st.audio(str(SAMPLE_AUDIO), format="audio/wav")
                use_sample = st.button("用样例锐评")
            else:
                st.info("还没有放入本地样例。")

        if uploaded_file is None and not use_sample:
            return

    is_curated_sample = False
    if use_sample:
        media_path = SAMPLE_AUDIO
        source_name = SAMPLE_AUDIO.name
        is_curated_sample = True
        track_name = "帰ろう（31 秒样例）"
        artist_name = "藤井風"
    else:
        media_path = save_upload_to_temp(uploaded_file)
        source_name = uploaded_file.name
        track_name = Path(source_name).stem

    sample_rate = 22050
    segment_seconds = 3.0

    loading_slot = st.empty()
    with loading_slot.container():
        render_loading()

    try:
        features = extract_features_from_media(
            media_path,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
        )
    except Exception as exc:
        loading_slot.empty()
        st.error(f"特征提取失败：{exc}")
        return

    loading_slot.empty()

    if features.empty:
        st.warning("没有提取到完整 3 秒切片。换个更长的音频，或者别只发一声响。")
        return

    if not artifacts_available():
        st.warning("模型工件还没生成。请先运行：python streamlit_demo\\build_demo_model_artifacts.py")
        return

    try:
        segment_predictions = predict_segments(features)
        genre_scores = aggregate_genre_prediction(segment_predictions)
    except Exception as exc:
        st.error(f"模型预测失败：{exc}")
        return

    st.markdown('<div class="radar-title">GTZAN 十类风格雷达</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_genre_radar(genre_scores), use_container_width=True)
    render_section_header("拆开听")
    render_feature_snapshot(features)
    st.markdown('<div class="chart-title">波形扫一眼</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_waveform(media_path, sample_rate), use_container_width=True)

    render_section_header("我锐评")

    reference = load_reference_data(str(REFERENCE_CSV)) if REFERENCE_CSV.exists() else features.copy()
    scores = style_scores(features, reference, genre_scores)
    if is_curated_sample:
        scores = apply_curated_sample_score(scores)
    predicted = str(genre_scores.iloc[0]["genre"])
    predicted_display = f"{GENRE_NAMES.get(predicted, predicted)} / {predicted}"
    verdict = verdict_text(scores, predicted, track_name)
    if is_curated_sample:
        predicted_display = "乐评特写：日式灵魂流行"
        verdict = curated_sample_verdict()

    result_left, result_right = st.columns([0.95, 1.05], gap="large")
    with result_left:
        score_genre_bars = render_score_genre_bars(genre_scores)
        st.markdown(
            f"""
            <div class="score-wrap">
              <div class="score-label">我锐评</div>
              <div><span class="score-number">{scores["total"]:.0f}</span><span class="score-suffix">分</span></div>
              <div class="small-note">模型风格判定：{predicted_display}</div>
              <div class="small-note">本歌气质：{axis_summary(scores)}</div>
              {"<div class='small-note'>固定样例：主结论走乐评特写，模型近邻见下图。</div>" if is_curated_sample else ""}
              {score_genre_bars}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_right:
        st.markdown('<div class="axis-inline-title">三条听感轴</div>', unsafe_allow_html=True)
        metric_bar("集百家长", "风格鲜明", scores["style_clear"])
        metric_bar("松弛耐听", "热烈直给", scores["energy"])
        metric_bar("低调内敛", "锋芒外放", scores["outward"])

    render_report_card(
        track_name=track_name,
        artist_name=artist_name,
        uploaded_name=source_name,
        score=scores["total"],
        predicted_display=predicted_display,
        scores=scores,
        verdict=verdict,
    )
    export_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in Path(source_name).stem)
    st.download_button(
        "导出锐评卡 PNG",
        data=report_card_png(
            track_name=track_name,
            artist_name=artist_name,
            uploaded_name=source_name,
            score=scores["total"],
            predicted_display=predicted_display,
            scores=scores,
            verdict=verdict,
        ),
        file_name=f"{export_stem}_ilisten_card.png",
        mime="image/png",
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title source-title">source</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="panel">
          <div class="source-lines">
            <span>数据</span>：GTZAN 十类音乐片段。<br>
            <span>特征</span>：节奏、响度、频谱、梅尔倒谱。<br>
            <span>模型</span>：支持向量机分类，给出前三类近邻。<br>
            <span>链路</span>：音频特征 → 十类风格分类 → iListen 锐评。<br>
            <span>分数</span>：主观展示指数，不负责给音乐封神。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("想看原始提取结果"):
        st.dataframe(features.head(30), use_container_width=True, hide_index=True)
        st.download_button(
            "下载特征 CSV",
            data=features.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{Path(source_name).stem}_features.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
