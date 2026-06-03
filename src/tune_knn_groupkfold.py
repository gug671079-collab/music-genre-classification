from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ===== 路径设置 =====
# 本文件位于 src 目录下，parents[1] 可以定位到项目根目录。
# 所有输入输出都使用项目内相对路径，不写死本地绝对路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path("data") / "processed" / "processed_features_3_sec_V2.csv"
OUTPUT_DIR = Path("logs") / "knn_tuning"
RESULTS_CSV_PATH = OUTPUT_DIR / "knn_tuning_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "knn_tuning_summary.txt"
TOP_PARAMS_IMAGE_PATH = OUTPUT_DIR / "knn_top_params.png"
CONFUSION_MATRIX_IMAGE_PATH = OUTPUT_DIR / "knn_confusion_matrix.png"
OOF_PREDICTIONS_PATH = OUTPUT_DIR / "knn_oof_predictions.csv"
K_CURVE_IMAGE_PATH = OUTPUT_DIR / "knn_k_curve.png"
HEATMAP_UNIFORM_IMAGE_PATH = OUTPUT_DIR / "knn_param_heatmap_uniform.png"
HEATMAP_DISTANCE_IMAGE_PATH = OUTPUT_DIR / "knn_param_heatmap_distance.png"
NORMALIZED_CONFUSION_MATRIX_IMAGE_PATH = (
    OUTPUT_DIR / "knn_confusion_matrix_normalized.png"
)
PER_CLASS_METRICS_IMAGE_PATH = OUTPUT_DIR / "knn_per_class_metrics.png"
FOLD_SCORES_CSV_PATH = OUTPUT_DIR / "knn_fold_scores.csv"
FOLD_SCORES_IMAGE_PATH = OUTPUT_DIR / "knn_fold_scores.png"

RANDOM_STATE = 42
N_SPLITS = 5


def require_matplotlib():
    """导入 matplotlib；缺少依赖时给出清晰错误提示。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "缺少 matplotlib，无法生成调参图像。请先安装 matplotlib，"
            "或在包含 matplotlib 的 Python 环境中运行本脚本。"
        ) from error

    return plt


def prepare_data(df):
    """检查必要字段，删除非模型字段，并只保留数值型音频特征。"""
    required_columns = ["label", "track_id"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError("数据缺少必要字段：" + ", ".join(missing_columns))

    non_model_columns = [
        "filename",
        "genre_from_filename",
        "track_id",
        "segment_id",
        "length",
    ]
    existing_non_model_columns = [
        column for column in non_model_columns if column in df.columns
    ]

    y = df["label"]
    groups = df["track_id"]
    X = df.drop(columns=existing_non_model_columns + ["label"])
    X = X.select_dtypes(include="number")

    if X.empty:
        raise ValueError("删除非模型字段后没有可用于训练的数值音频特征。")

    return X, y, groups, existing_non_model_columns


def build_knn_pipeline():
    """构建 KNN 调参使用的 Pipeline，避免交叉验证中的标准化泄漏。"""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]
    )


def get_param_grid():
    """定义 KNN 参数搜索范围。"""
    return {
        "model__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 41],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }


def run_grid_search(X, y, groups):
    """使用 GroupKFold 和 GridSearchCV 搜索 KNN 最优参数。"""
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }
    grid_search = GridSearchCV(
        estimator=build_knn_pipeline(),
        param_grid=get_param_grid(),
        cv=GroupKFold(n_splits=N_SPLITS),
        scoring=scoring,
        refit="f1_macro",
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(X, y, groups=groups)

    return grid_search


def get_top_10_results(results_df):
    """提取 Macro-F1 排名前 10 的参数组合。"""
    top_columns = [
        "rank_test_f1_macro",
        "param_model__n_neighbors",
        "param_model__weights",
        "param_model__p",
        "mean_test_f1_macro",
        "std_test_f1_macro",
        "mean_test_accuracy",
        "mean_test_f1_weighted",
    ]

    return (
        results_df.sort_values("rank_test_f1_macro")
        .loc[:, top_columns]
        .head(10)
        .reset_index(drop=True)
    )


def plot_top_params(top_10_results, output_path):
    """绘制排名前 10 的 KNN 参数组合 Macro-F1 横向柱状图。"""
    plt = require_matplotlib()
    plot_df = top_10_results.sort_values("mean_test_f1_macro", ascending=True)
    labels = [
        f"k={row.param_model__n_neighbors}, "
        f"w={row.param_model__weights}, p={row.param_model__p}"
        for row in plot_df.itertuples(index=False)
    ]

    figure, axis = plt.subplots(figsize=(11, 7))
    axis.barh(labels, plot_df["mean_test_f1_macro"])
    axis.set_title("Top 10 KNN Parameter Combinations by GroupKFold Macro F1")
    axis.set_xlabel("Mean Test Macro-F1")
    axis.set_ylabel("Parameter Combination")
    axis.grid(axis="x", linestyle="--", alpha=0.4)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_k_curve(results_df, output_path):
    """绘制 n_neighbors 对 GroupKFold Macro-F1 的影响曲线。"""
    plt = require_matplotlib()
    figure, axis = plt.subplots(figsize=(10, 6))

    for (weights, p_value), group_df in results_df.groupby(
        ["param_model__weights", "param_model__p"]
    ):
        line_df = group_df.sort_values("param_model__n_neighbors")
        axis.errorbar(
            line_df["param_model__n_neighbors"],
            line_df["mean_test_f1_macro"],
            yerr=line_df["std_test_f1_macro"],
            marker="o",
            capsize=3,
            label=f"weights={weights}, p={p_value}",
        )

    axis.set_title("KNN n_neighbors vs GroupKFold Macro F1")
    axis.set_xlabel("n_neighbors")
    axis.set_ylabel("mean_test_f1_macro")
    axis.legend()
    axis.grid(linestyle="--", alpha=0.4)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_param_heatmap(results_df, weights, output_path):
    """绘制指定 weights 下 n_neighbors 与 p 对 Macro-F1 的影响热力图。"""
    plt = require_matplotlib()
    heatmap_df = (
        results_df.loc[results_df["param_model__weights"] == weights]
        .pivot(
            index="param_model__n_neighbors",
            columns="param_model__p",
            values="mean_test_f1_macro",
        )
        .sort_index()
    )

    figure, axis = plt.subplots(figsize=(7, 8))
    image = axis.imshow(heatmap_df.values, cmap="Blues", aspect="auto")
    figure.colorbar(image, ax=axis, label="mean_test_f1_macro")

    axis.set_title(f"KNN Macro F1 Heatmap - weights={weights}")
    axis.set_xlabel("p")
    axis.set_ylabel("n_neighbors")
    axis.set_xticks(range(len(heatmap_df.columns)))
    axis.set_yticks(range(len(heatmap_df.index)))
    axis.set_xticklabels(heatmap_df.columns)
    axis.set_yticklabels(heatmap_df.index)

    for row_index in range(heatmap_df.shape[0]):
        for column_index in range(heatmap_df.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{heatmap_df.values[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                color="black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_param_heatmaps(results_df):
    """分别保存 uniform 和 distance 两种 weights 的参数热力图。"""
    plot_param_heatmap(
        results_df,
        weights="uniform",
        output_path=PROJECT_ROOT / HEATMAP_UNIFORM_IMAGE_PATH,
    )
    plot_param_heatmap(
        results_df,
        weights="distance",
        output_path=PROJECT_ROOT / HEATMAP_DISTANCE_IMAGE_PATH,
    )


def generate_oof_predictions_and_fold_scores(X, y, groups, best_params):
    """用最优参数重新执行 GroupKFold，生成 OOF 预测和每折指标。"""
    splitter = GroupKFold(n_splits=N_SPLITS)
    base_model = build_knn_pipeline()
    base_model.set_params(**best_params)

    y_oof_pred = pd.Series(index=y.index, dtype="object")
    fold_scores = []

    for fold_number, (train_index, validation_index) in enumerate(
        splitter.split(X, y, groups=groups),
        start=1,
    ):
        fold_model = clone(base_model)
        X_train_fold = X.iloc[train_index]
        X_validation_fold = X.iloc[validation_index]
        y_train_fold = y.iloc[train_index]
        y_validation_fold = y.iloc[validation_index]
        train_groups = groups.iloc[train_index]
        validation_groups = groups.iloc[validation_index]

        fold_model.fit(X_train_fold, y_train_fold)
        validation_pred = fold_model.predict(X_validation_fold)
        validation_original_index = X.index[validation_index]
        y_oof_pred.loc[validation_original_index] = validation_pred
        fold_scores.append(
            {
                "fold": fold_number,
                "train_sample_count": len(train_index),
                "validation_sample_count": len(validation_index),
                "train_track_count": train_groups.nunique(),
                "validation_track_count": validation_groups.nunique(),
                "accuracy": accuracy_score(y_validation_fold, validation_pred),
                "macro_precision": precision_score(
                    y_validation_fold,
                    validation_pred,
                    average="macro",
                    zero_division=0,
                ),
                "macro_recall": recall_score(
                    y_validation_fold,
                    validation_pred,
                    average="macro",
                    zero_division=0,
                ),
                "f1_macro": f1_score(
                    y_validation_fold,
                    validation_pred,
                    average="macro",
                    zero_division=0,
                ),
                "f1_weighted": f1_score(
                    y_validation_fold,
                    validation_pred,
                    average="weighted",
                    zero_division=0,
                ),
            }
        )

    oof_predictions = pd.DataFrame(
        {
            "sample_index": X.index,
            "track_id": groups,
            "y_true": y,
            "y_pred": y_oof_pred,
        }
    )

    if oof_predictions["y_pred"].isnull().any():
        raise ValueError("OOF 预测存在缺失值，请检查 GroupKFold 划分过程。")

    return oof_predictions, pd.DataFrame(fold_scores)


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """基于 OOF 预测绘制混淆矩阵。"""
    plt = require_matplotlib()
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_title("KNN GroupKFold OOF Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                matrix[row_index, column_index],
                ha="center",
                va="center",
                color="black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_normalized_confusion_matrix(y_true, y_pred, labels, output_path):
    """基于 OOF 预测绘制按真实类别归一化的混淆矩阵。"""
    plt = require_matplotlib()
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )

    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(normalized_matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Row-normalized ratio")

    axis.set_title("KNN GroupKFold OOF Normalized Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)

    for row_index in range(normalized_matrix.shape[0]):
        for column_index in range(normalized_matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{normalized_matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def compute_per_class_metrics(y_true, y_pred, labels):
    """基于 OOF 预测计算每个类别的 precision、recall、f1-score 和 support。"""
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    rows = []

    for label in labels:
        metrics = report_dict[label]
        rows.append(
            {
                "label": label,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1-score": metrics["f1-score"],
                "support": int(metrics["support"]),
            }
        )

    return pd.DataFrame(rows)


def plot_per_class_metrics(per_class_metrics, output_path):
    """绘制每个类别的 precision、recall、f1-score 对比柱状图。"""
    plt = require_matplotlib()
    labels = per_class_metrics["label"].tolist()
    x_positions = np.arange(len(labels))
    width = 0.25

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(
        x_positions - width,
        per_class_metrics["precision"],
        width=width,
        label="precision",
    )
    axis.bar(
        x_positions,
        per_class_metrics["recall"],
        width=width,
        label="recall",
    )
    axis.bar(
        x_positions + width,
        per_class_metrics["f1-score"],
        width=width,
        label="f1-score",
    )

    axis.set_title("KNN GroupKFold OOF Per-class Metrics")
    axis.set_xlabel("Class Label")
    axis.set_ylabel("Score")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.4)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_fold_scores(fold_scores, output_path):
    """绘制最优参数下 5 折 OOF Macro-F1 波动。"""
    plt = require_matplotlib()
    mean_f1_macro = fold_scores["f1_macro"].mean()

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        fold_scores["fold"],
        fold_scores["f1_macro"],
        marker="o",
        label="Fold Macro-F1",
    )
    axis.axhline(
        mean_f1_macro,
        color="red",
        linestyle="--",
        label=f"Mean Macro-F1 = {mean_f1_macro:.3f}",
    )

    axis.set_title("KNN Best Params GroupKFold Fold Macro F1")
    axis.set_xlabel("Fold")
    axis.set_ylabel("f1_macro")
    axis.set_xticks(fold_scores["fold"])
    axis.legend()
    axis.grid(linestyle="--", alpha=0.4)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def get_top_confusions(y_true, y_pred, labels, top_n=10):
    """提取 OOF 混淆矩阵中误判数量最高的类别对。"""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []

    for true_index, true_label in enumerate(labels):
        row_total = matrix[true_index].sum()
        for pred_index, predicted_label in enumerate(labels):
            if true_index == pred_index:
                continue

            count = int(matrix[true_index, pred_index])
            if count == 0:
                continue

            rows.append(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                    "row_normalized_ratio": count / row_total if row_total else 0,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "true_label",
                "predicted_label",
                "count",
                "row_normalized_ratio",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["count", "row_normalized_ratio"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def format_class_distribution(y):
    """整理类别分布文本。"""
    return y.value_counts().sort_index().to_string()


def build_summary_log(
    y,
    groups,
    dropped_columns,
    param_grid,
    best_params,
    best_row,
    top_10_results,
    oof_predictions,
    fold_scores,
    per_class_metrics,
    top_confusions,
):
    """生成 KNN GroupKFold 调参实验总结日志。"""
    oof_accuracy = accuracy_score(
        oof_predictions["y_true"],
        oof_predictions["y_pred"],
    )
    oof_macro_f1 = f1_score(
        oof_predictions["y_true"],
        oof_predictions["y_pred"],
        average="macro",
        zero_division=0,
    )
    oof_weighted_f1 = f1_score(
        oof_predictions["y_true"],
        oof_predictions["y_pred"],
        average="weighted",
        zero_division=0,
    )

    summary_lines = [
        "KNN GroupKFold 调参实验",
        "=" * 50,
        "",
        "一、数据路径",
        f"读取数据：{DATA_PATH.as_posix()}",
        "",
        "二、数据规模",
        f"样本数量：{len(y)}",
        f"track_id 数量：{groups.nunique()}",
        "",
        "三、类别分布",
        format_class_distribution(y),
        "",
        "四、字段处理",
        "目标变量：label",
        "分组字段：track_id",
        "删除的非模型字段：",
        ", ".join(dropped_columns),
        "",
        "五、验证方式",
        f"使用的验证方式：GroupKFold(n_splits={N_SPLITS})",
        "主要评价指标：Macro-F1",
        "",
        "六、参数搜索范围",
        str(param_grid),
        "p=1 表示 Manhattan distance；p=2 表示 Euclidean distance。",
        "",
        "七、最优参数与分数",
        f"最优参数：{best_params}",
        f"最优 mean_test_f1_macro：{best_row['mean_test_f1_macro']:.4f}",
        f"对应 std_test_f1_macro：{best_row['std_test_f1_macro']:.4f}",
        f"对应 mean_test_accuracy：{best_row['mean_test_accuracy']:.4f}",
        f"对应 mean_test_f1_weighted：{best_row['mean_test_f1_weighted']:.4f}",
        "",
        "八、排名前 10 的参数组合",
        top_10_results.to_string(index=False),
        "",
        "九、OOF 预测结果",
        f"OOF Accuracy：{oof_accuracy:.4f}",
        f"OOF Macro-F1：{oof_macro_f1:.4f}",
        f"OOF Weighted-F1：{oof_weighted_f1:.4f}",
        "",
        "十、每一折结果",
        fold_scores.to_string(index=False),
        "",
        "十一、每个类别的 OOF 指标",
        per_class_metrics.to_string(index=False),
        "",
        "十二、最容易混淆的前 10 个类别对",
        top_confusions.to_string(index=False) if not top_confusions.empty else "无",
        "",
        "十三、与 Baseline V2 结果的比较说明",
        "本实验结果不能直接和 Baseline V2 的片段级 train_test_split 结果简单比较。",
        "Baseline V2 使用 train_test_split(stratify=y)，可能存在同一首音乐的不同片段跨训练集和测试集的问题。",
        "本实验使用 GroupKFold，并以 track_id 分组，评估方式更严格，更接近未见歌曲的泛化表现。",
        "本实验应与 Baseline V3 GroupKFold 在同一评估框架下比较。Baseline V3 是固定参数 KNN，本实验是在相同 GroupKFold 设置下进行 KNN 参数搜索，因此两者可以用于展示 KNN 调参带来的性能变化。Baseline V2 使用片段级 train_test_split，可能结果偏乐观，不作为后续正式模型对比基准。",
        "",
        "十四、输出文件列表",
        RESULTS_CSV_PATH.as_posix(),
        SUMMARY_PATH.as_posix(),
        TOP_PARAMS_IMAGE_PATH.as_posix(),
        CONFUSION_MATRIX_IMAGE_PATH.as_posix(),
        OOF_PREDICTIONS_PATH.as_posix(),
        K_CURVE_IMAGE_PATH.as_posix(),
        HEATMAP_UNIFORM_IMAGE_PATH.as_posix(),
        HEATMAP_DISTANCE_IMAGE_PATH.as_posix(),
        NORMALIZED_CONFUSION_MATRIX_IMAGE_PATH.as_posix(),
        PER_CLASS_METRICS_IMAGE_PATH.as_posix(),
        FOLD_SCORES_CSV_PATH.as_posix(),
        FOLD_SCORES_IMAGE_PATH.as_posix(),
        "",
        "实验说明：本脚本只进行 KNN 调参，统一使用 GroupKFold 避免同一首音乐的片段跨训练折和验证折。",
        "实验完成。",
    ]

    return "\n".join(summary_lines)


def main():
    """执行 KNN GroupKFold 调参实验，并保存结果、图像和日志。"""
    data_file = PROJECT_ROOT / DATA_PATH
    output_dir = PROJECT_ROOT / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file, encoding="utf-8")
    X, y, groups, dropped_columns = prepare_data(df)

    grid_search = run_grid_search(X, y, groups)
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(
        PROJECT_ROOT / RESULTS_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    top_10_results = get_top_10_results(results_df)
    plot_top_params(top_10_results, PROJECT_ROOT / TOP_PARAMS_IMAGE_PATH)
    plot_k_curve(results_df, PROJECT_ROOT / K_CURVE_IMAGE_PATH)
    plot_param_heatmaps(results_df)

    oof_predictions, fold_scores = generate_oof_predictions_and_fold_scores(
        X=X,
        y=y,
        groups=groups,
        best_params=grid_search.best_params_,
    )
    oof_predictions.to_csv(
        PROJECT_ROOT / OOF_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8",
    )
    fold_scores.to_csv(
        PROJECT_ROOT / FOLD_SCORES_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    labels = sorted(y.unique())
    plot_confusion_matrix(
        y_true=oof_predictions["y_true"],
        y_pred=oof_predictions["y_pred"],
        labels=labels,
        output_path=PROJECT_ROOT / CONFUSION_MATRIX_IMAGE_PATH,
    )
    plot_normalized_confusion_matrix(
        y_true=oof_predictions["y_true"],
        y_pred=oof_predictions["y_pred"],
        labels=labels,
        output_path=PROJECT_ROOT / NORMALIZED_CONFUSION_MATRIX_IMAGE_PATH,
    )
    per_class_metrics = compute_per_class_metrics(
        y_true=oof_predictions["y_true"],
        y_pred=oof_predictions["y_pred"],
        labels=labels,
    )
    plot_per_class_metrics(
        per_class_metrics,
        PROJECT_ROOT / PER_CLASS_METRICS_IMAGE_PATH,
    )
    plot_fold_scores(fold_scores, PROJECT_ROOT / FOLD_SCORES_IMAGE_PATH)
    top_confusions = get_top_confusions(
        y_true=oof_predictions["y_true"],
        y_pred=oof_predictions["y_pred"],
        labels=labels,
    )

    best_row = results_df.loc[grid_search.best_index_]
    summary_log = build_summary_log(
        y=y,
        groups=groups,
        dropped_columns=dropped_columns,
        param_grid=get_param_grid(),
        best_params=grid_search.best_params_,
        best_row=best_row,
        top_10_results=top_10_results,
        oof_predictions=oof_predictions,
        fold_scores=fold_scores,
        per_class_metrics=per_class_metrics,
        top_confusions=top_confusions,
    )
    (PROJECT_ROOT / SUMMARY_PATH).write_text(summary_log, encoding="utf-8")

    output_files = [
        RESULTS_CSV_PATH,
        SUMMARY_PATH,
        TOP_PARAMS_IMAGE_PATH,
        CONFUSION_MATRIX_IMAGE_PATH,
        OOF_PREDICTIONS_PATH,
        K_CURVE_IMAGE_PATH,
        HEATMAP_UNIFORM_IMAGE_PATH,
        HEATMAP_DISTANCE_IMAGE_PATH,
        NORMALIZED_CONFUSION_MATRIX_IMAGE_PATH,
        PER_CLASS_METRICS_IMAGE_PATH,
        FOLD_SCORES_CSV_PATH,
        FOLD_SCORES_IMAGE_PATH,
    ]

    print("KNN GroupKFold 调参实验完成。输出文件如下：")
    for output_file in output_files:
        print(output_file.as_posix())


if __name__ == "__main__":
    main()
