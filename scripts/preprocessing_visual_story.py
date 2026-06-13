from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_features_3_sec.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "preprocessing_visualizations"

LABEL_ORDER = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

FEATURE_NAME_ZH = {
    "length": "片段长度",
    "chroma_stft_mean": "色度均值",
    "rms_mean": "能量RMS",
    "spectral_centroid_mean": "频谱质心",
    "spectral_bandwidth_mean": "频谱带宽",
    "rolloff_mean": "频谱滚降",
    "zero_crossing_rate_mean": "过零率",
    "harmony_mean": "和声成分",
    "perceptr_mean": "感知成分",
    "tempo": "节奏速度",
    "mfcc1_mean": "MFCC1",
    "mfcc2_mean": "MFCC2",
    "mfcc3_mean": "MFCC3",
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    font_path_candidates = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for font_path in font_path_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_figure(name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, dpi=240, bbox_inches="tight")
    plt.close()


def feature_family(column: str) -> str:
    if column == "length":
        return "片段长度"
    if column.startswith("chroma"):
        return "色度特征"
    if column.startswith("rms"):
        return "能量特征"
    if column.startswith("spectral") or column.startswith("rolloff"):
        return "频谱特征"
    if column.startswith("zero_crossing"):
        return "过零率特征"
    if column.startswith("harmony") or column.startswith("perceptr"):
        return "和声/感知特征"
    if column == "tempo":
        return "节奏特征"
    if column.startswith("mfcc"):
        return "MFCC特征"
    return "其他特征"


def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA_PATH)
    numeric_features = df.drop(columns=["label"]).select_dtypes(include=[np.number]).columns.tolist()
    return df, numeric_features


def plot_preprocessing_overview(df: pd.DataFrame, numeric_features: list[str]) -> dict[str, int]:
    label_counts = df["label"].value_counts().reindex(LABEL_ORDER)
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    feature_duplicate_rows = int(df.drop(columns=["label"]).duplicated().sum())

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.65], wspace=0.32, hspace=0.34)

    cards = [
        ("样本总量", f"{len(df):,}", "3 秒音频切片"),
        ("音乐流派", f"{df['label'].nunique()}", "GTZAN 十分类"),
        ("数值特征", f"{len(numeric_features)}", "label 不计入"),
        ("缺失值", f"{missing_values}", "无缺失可直接建模"),
    ]
    card_colors = ["#2563eb", "#059669", "#7c3aed", "#dc2626"]
    for index, (title, value, note) in enumerate(cards):
        ax = fig.add_subplot(grid[0, index])
        ax.set_facecolor("#f8fafc")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#d9dee7")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.06, 0.78, title, transform=ax.transAxes, fontsize=17, weight="bold", color="#111827")
        ax.text(0.06, 0.34, value, transform=ax.transAxes, fontsize=34, weight="bold", color=card_colors[index])
        ax.text(0.06, 0.12, note, transform=ax.transAxes, fontsize=12, color="#64748b")

    ax_bar = fig.add_subplot(grid[1, :3])
    bars = ax_bar.bar(label_counts.index, label_counts.values, color="#3b82f6", edgecolor="#1e3a8a", linewidth=0.8)
    ax_bar.bar_label(bars, fontsize=10, padding=3)
    ax_bar.set_title("各音乐流派样本数量", fontsize=18, weight="bold", pad=14)
    ax_bar.set_ylabel("样本数")
    ax_bar.set_ylim(0, int(label_counts.max()) + 95)
    ax_bar.tick_params(axis="x", rotation=32)
    ax_bar.grid(axis="y", color="#e5e7eb")

    ax_dup = fig.add_subplot(grid[1, 3])
    dup_labels = ["完整重复行", "特征重复行"]
    dup_values = [duplicate_rows, feature_duplicate_rows]
    ax_dup.barh(dup_labels, dup_values, color=["#f59e0b", "#ef4444"])
    ax_dup.set_title("重复样本检查", fontsize=18, weight="bold", pad=14)
    ax_dup.set_xlabel("行数")
    ax_dup.set_xlim(0, max(dup_values) * 1.38 if max(dup_values) else 1)
    ax_dup.set_yticks([])
    for y, (label, value) in enumerate(zip(dup_labels, dup_values)):
        ax_dup.text(5, y, label, va="center", ha="left", fontsize=12, color="#111827")
        ax_dup.text(value + max(dup_values) * 0.04, y, str(value), va="center", fontsize=12)
    ax_dup.grid(axis="x", color="#e5e7eb")

    fig.suptitle("数据预处理概览：规模、均衡性与质量检查", fontsize=23, weight="bold", y=1.02)
    save_figure("06_preprocessing_overview.png")
    return {
        "sample_count": len(df),
        "genre_count": int(df["label"].nunique()),
        "feature_count": len(numeric_features),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "feature_duplicate_rows": feature_duplicate_rows,
        "class_min": int(label_counts.min()),
        "class_max": int(label_counts.max()),
    }


def plot_feature_family_structure(numeric_features: list[str]) -> pd.Series:
    family_order = [
        "MFCC特征",
        "频谱特征",
        "和声/感知特征",
        "色度特征",
        "能量特征",
        "过零率特征",
        "节奏特征",
        "片段长度",
    ]
    family_counts = pd.Series([feature_family(column) for column in numeric_features]).value_counts()
    family_counts = family_counts.reindex(family_order).dropna().astype(int)

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.22)

    ax_donut = fig.add_subplot(grid[0, 0])
    colors = sns.color_palette("Set2", n_colors=len(family_counts))
    wedges, _ = ax_donut.pie(
        family_counts.values,
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
    )
    ax_donut.text(0, 0.06, str(int(family_counts.sum())), ha="center", va="center", fontsize=40, weight="bold")
    ax_donut.text(0, -0.17, "个数值特征", ha="center", va="center", fontsize=14, color="#64748b")
    ax_donut.set_title("特征家族占比", fontsize=18, weight="bold", pad=16)

    ax_bar = fig.add_subplot(grid[0, 1])
    y = np.arange(len(family_counts))
    ax_bar.barh(y, family_counts.values, color=colors, edgecolor="#334155", linewidth=0.6)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(family_counts.index)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("特征数量")
    ax_bar.set_title("音频特征类型结构", fontsize=18, weight="bold", pad=16)
    ax_bar.grid(axis="x", color="#e5e7eb")
    for index, value in enumerate(family_counts.values):
        ax_bar.text(value + 0.55, index, str(value), va="center", fontsize=13)
    ax_bar.set_xlim(0, max(family_counts.values) + 5)

    fig.suptitle("模型输入特征结构：从单一音频到多维描述", fontsize=23, weight="bold", y=1.02)
    save_figure("07_feature_family_structure.png")
    return family_counts


def plot_feature_scale(df: pd.DataFrame) -> pd.DataFrame:
    selected = [
        "rms_mean",
        "zero_crossing_rate_mean",
        "chroma_stft_mean",
        "mfcc1_mean",
        "tempo",
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "rolloff_mean",
    ]
    rows = []
    for column in selected:
        values = df[column].astype(float)
        rows.append(
            {
                "feature": column,
                "display": FEATURE_NAME_ZH.get(column, column),
                "std": float(values.std()),
                "p05": float(values.quantile(0.05)),
                "p95": float(values.quantile(0.95)),
                "median": float(values.median()),
            }
        )
    scale_df = pd.DataFrame(rows).sort_values("std", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.barh(scale_df["display"], scale_df["std"], color="#0f766e", edgecolor="#064e3b", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("原始标准差（对数坐标）")
    ax.set_title("代表特征原始尺度差异：标准化处理的必要性", fontsize=21, weight="bold", pad=16)
    ax.grid(axis="x", color="#e5e7eb")
    for y, value in enumerate(scale_df["std"]):
        ax.text(value * 1.12, y, f"{value:.3g}", va="center", fontsize=11)
    ax.text(
        0.01,
        -0.16,
        "注：频谱类特征、节奏速度与 RMS/过零率等特征处在完全不同的数值尺度，直接输入距离或间隔类模型会放大大尺度特征的影响。",
        transform=ax.transAxes,
        fontsize=11,
        color="#475569",
    )
    save_figure("08_feature_scale_before_standardization.png")
    return scale_df


def plot_genre_fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    selected = [
        "rms_mean",
        "chroma_stft_mean",
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "rolloff_mean",
        "zero_crossing_rate_mean",
        "tempo",
        "mfcc1_mean",
        "mfcc2_mean",
    ]
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[selected]), columns=selected)
    scaled["label"] = df["label"].values
    fingerprint = scaled.groupby("label")[selected].mean().reindex(LABEL_ORDER)
    fingerprint.columns = [FEATURE_NAME_ZH.get(column, column) for column in fingerprint.columns]

    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    sns.heatmap(
        fingerprint,
        ax=ax,
        cmap="vlag",
        center=0,
        linewidths=0.55,
        linecolor="white",
        cbar_kws={"label": "标准化后类别均值"},
    )
    ax.set_title("流派特征指纹：不同音乐风格的平均音频画像", fontsize=21, weight="bold", pad=16)
    ax.set_xlabel("代表音频特征")
    ax.set_ylabel("音乐流派")
    ax.tick_params(axis="x", rotation=32)
    ax.tick_params(axis="y", rotation=0)
    save_figure("09_genre_feature_fingerprint.png")
    return fingerprint


def plot_pca_distribution(df: pd.DataFrame, numeric_features: list[str]) -> tuple[float, float]:
    X = df[numeric_features].astype(float)
    y = df["label"]
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": points[:, 0], "PC2": points[:, 1], "label": y})
    sample = pca_df.groupby("label", group_keys=False).sample(n=220, random_state=42)
    centroids = pca_df.groupby("label")[["PC1", "PC2"]].mean().reindex(LABEL_ORDER)

    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    palette = dict(zip(LABEL_ORDER, sns.color_palette("tab10", n_colors=len(LABEL_ORDER))))
    sns.scatterplot(
        data=sample,
        x="PC1",
        y="PC2",
        hue="label",
        hue_order=LABEL_ORDER,
        palette=palette,
        s=30,
        alpha=0.62,
        linewidth=0,
        ax=ax,
    )
    for label, row in centroids.iterrows():
        ax.scatter(row["PC1"], row["PC2"], s=130, marker="X", color=palette[label], edgecolor="white", linewidth=1.2)
        ax.text(row["PC1"] + 0.12, row["PC2"] + 0.12, label, fontsize=10, weight="bold", color="#111827")

    explained = pca.explained_variance_ratio_ * 100
    ax.set_title("PCA 流派分布：有聚集，也有明显重叠", fontsize=21, weight="bold", pad=16)
    ax.set_xlabel(f"PC1（解释方差 {explained[0]:.1f}%）")
    ax.set_ylabel(f"PC2（解释方差 {explained[1]:.1f}%）")
    ax.grid(color="#e5e7eb")
    ax.legend(title="音乐流派", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.text(
        0.01,
        -0.14,
        "注：每类抽样 220 个片段绘制散点，X 标记为该类别在二维 PCA 空间中的中心位置。",
        transform=ax.transAxes,
        fontsize=11,
        color="#475569",
    )
    save_figure("10_pca_genre_distribution_enhanced.png")
    return float(explained[0]), float(explained[1])


def write_report_text(
    overview: dict[str, int],
    family_counts: pd.Series,
    scale_df: pd.DataFrame,
    fingerprint: pd.DataFrame,
    pca_explained: tuple[float, float],
) -> None:
    top_family = family_counts.idxmax()
    top_family_count = int(family_counts.max())
    largest_scale = scale_df.sort_values("std", ascending=False).iloc[0]
    smallest_scale = scale_df.sort_values("std", ascending=True).iloc[0]
    fingerprint_abs = fingerprint.abs().mean(axis=1).sort_values(ascending=False)
    strongest_profile_genre = str(fingerprint_abs.index[0])

    text = f"""# 数据预处理可视化报告文案

本文件对应 `docs/preprocessing_visualizations` 中新增的 5 张数据预处理与数据分析图表。文字可直接作为技术报告“数据描述与预处理”或“数据分析与可视化”章节的素材使用。

## 图 06 数据预处理概览图

**图表目的：**  
该图用于概括当前特征数据表的基本规模、标签结构和数据质量检查结果，帮助说明数据是否具备进入后续建模阶段的基本条件。

**结果解读：**  
当前数据表共包含 {overview["sample_count"]} 条 3 秒音频切片样本，覆盖 {overview["genre_count"]} 类音乐风格。除 `label` 外，数据中包含 {overview["feature_count"]} 个数值型特征。各类别样本数分布在 {overview["class_min"]} 至 {overview["class_max"]} 之间，整体接近均衡，因此后续模型训练不需要优先处理严重类别不平衡问题。缺失值检查结果为 {overview["missing_values"]}，说明当前特征矩阵不存在缺失字段，可以直接进入统计分析和模型训练流程。同时，完整重复行检查发现 {overview["duplicate_rows"]} 条重复样本，特征层面重复行检查发现 {overview["feature_duplicate_rows"]} 条重复样本，这也提示后续实验需要关注去重策略、切片来源以及基于歌曲级别的划分方式，避免重复或同源样本对模型评估造成偏乐观影响。

**可放进报告的正式段落：**  
在数据预处理阶段，首先对 GTZAN 3 秒音频切片特征数据进行整体质量检查。结果显示，数据集共包含 {overview["sample_count"]} 条样本，覆盖 blues、classical、country、disco、hiphop、jazz、metal、pop、reggae、rock 共 {overview["genre_count"]} 个音乐风格类别。各类别样本数基本接近，说明数据整体较为均衡，Accuracy 与 Macro F1 等评价指标具有较好的参考意义。缺失值统计结果为 {overview["missing_values"]}，表明当前特征表不存在缺失字段。与此同时，重复值检查发现仍存在一定数量的重复记录，说明后续模型实验中需要进一步关注去重处理和基于歌曲来源的分组划分，以降低同源样本或重复样本对评估结果的影响。

## 图 07 音频特征类型结构图

**图表目的：**  
该图用于展示模型输入特征的组成结构，说明本项目并不是依赖单一音频指标进行分类，而是综合使用多类音频描述特征。

**结果解读：**  
从特征家族统计可以看出，当前数据表中的特征主要由 {top_family} 构成，该类特征共有 {top_family_count} 个，是输入特征体系中占比最高的部分。除此之外，数据还包含频谱、色度、能量、过零率、节奏以及和声/感知类特征。这些特征分别描述了音乐在音色、频率分布、能量强弱、节奏速度和短时谱包络等方面的信息，能够从多个角度表征音乐风格。

**可放进报告的正式段落：**  
为了更清晰地理解模型输入，本项目按照音频特征的含义对数值字段进行归类。结果显示，当前特征体系由 MFCC、频谱、色度、能量、过零率、节奏以及和声/感知等多类特征共同构成。其中，MFCC 特征数量最多，能够较好描述音频的短时谱包络和音色结构；频谱质心、频谱带宽和频谱滚降点等频谱特征则反映声音频率分布和明亮度；RMS、过零率和 tempo 分别提供能量、波形变化和节奏速度信息。多类型特征的组合为音乐风格分类提供了较完整的音频描述基础。

## 图 08 标准化必要性图

**图表目的：**  
该图用于解释为什么模型训练前需要进行标准化处理，尤其是对 KNN、SVM 这类受距离或间隔影响明显的模型。

**结果解读：**  
不同音频特征的原始数值尺度差异明显。例如，{largest_scale["display"]} 的原始标准差约为 {largest_scale["std"]:.3g}，而 {smallest_scale["display"]} 的原始标准差约为 {smallest_scale["std"]:.3g}。如果直接将这些特征输入模型，数值范围较大的频谱类或节奏类特征可能在距离计算和分类边界学习中占据更大权重，而数值范围较小的 RMS、色度或过零率特征容易被弱化。因此，在后续建模中使用 `StandardScaler` 对特征进行标准化是必要步骤。

**可放进报告的正式段落：**  
音频特征之间存在显著的量纲和数值尺度差异。频谱质心、频谱带宽、频谱滚降点和节奏速度等特征的数值范围通常远大于 RMS、过零率和色度特征。若不进行标准化处理，KNN 模型中的距离计算和 SVM 模型中的间隔优化都可能被大尺度特征主导，从而削弱其他有效特征的作用。基于这一点，本文在模型训练阶段将 `StandardScaler` 纳入统一的 `Pipeline` 流程，使每个特征在训练过程中被转换到相近尺度，从而提升不同特征对分类任务的公平贡献。

## 图 09 流派特征指纹图

**图表目的：**  
该图用于展示不同音乐风格在代表性音频特征上的平均差异，形成类似“音频画像”的对比结果。

**结果解读：**  
流派特征指纹图将代表性特征标准化后按类别求平均值。颜色偏红表示该流派在某一特征上的平均水平高于整体平均，颜色偏蓝表示低于整体平均。图中可以看到，不同流派在能量、频谱、过零率、节奏和 MFCC 等维度上呈现出不同组合模式，其中 {strongest_profile_genre} 的整体特征偏离程度相对更明显。这说明音乐风格差异可以通过多维音频特征得到一定刻画，为监督学习分类提供了数据基础。

**可放进报告的正式段落：**  
为进一步观察不同音乐风格的特征差异，本文选取 RMS、色度、频谱质心、频谱带宽、频谱滚降点、过零率、节奏速度和 MFCC 等代表性特征，经过标准化后计算各类别的平均特征画像。结果表明，不同音乐风格在多个音频维度上具有不同的组合模式，说明音乐风格标签并非随机分布，而是能够在数值特征空间中体现出一定规律。与此同时，不同类别之间并不是在所有特征上都完全分离，这也说明后续分类模型需要综合多个特征维度进行判断。

## 图 10 PCA 流派分布增强图

**图表目的：**  
该图用于将 58 维音频特征压缩到二维空间，直观观察不同音乐风格样本之间的聚集和重叠情况。

**结果解读：**  
PCA 前两个主成分分别解释了 {pca_explained[0]:.1f}% 和 {pca_explained[1]:.1f}% 的方差。图中可以观察到，部分类别在二维空间中存在一定聚集趋势，说明音频特征确实包含与音乐风格相关的信息；但不同类别之间也存在明显重叠，尤其是风格边界接近的类别并不能仅通过二维线性投影完全分开。这说明音乐风格分类任务具有一定可分性，同时也存在较强混叠和非线性边界问题。

**可放进报告的正式段落：**  
为了从整体上观察样本在高维特征空间中的分布情况，本文使用 PCA 将 58 维数值特征降至二维。结果显示，部分音乐风格在低维空间中具有一定聚集趋势，说明当前音频特征能够反映风格差异；但不同类别之间仍存在明显交叠，表明该任务并不是简单线性可分问题。这一现象也解释了为什么后续模型训练中需要比较 KNN、SVM-RBF 和 RandomForest 等不同模型，并使用能够刻画非线性边界的模型进一步提升分类效果。
"""

    (OUTPUT_DIR / "README_报告文案.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    df, numeric_features = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overview = plot_preprocessing_overview(df, numeric_features)
    family_counts = plot_feature_family_structure(numeric_features)
    scale_df = plot_feature_scale(df)
    fingerprint = plot_genre_fingerprint(df)
    pca_explained = plot_pca_distribution(df, numeric_features)
    write_report_text(overview, family_counts, scale_df, fingerprint, pca_explained)

    print(f"Saved preprocessing visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
