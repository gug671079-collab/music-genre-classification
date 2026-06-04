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
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ===== 路径设置 =====
# 本文件位于 src 目录下，parents[1] 可以定位到项目根目录。
# 所有输入输出都使用项目内相对路径，不写死任何本地绝对路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path("data") / "processed" / "processed_features_3_sec_V2.csv"
OUTPUT_DIR = Path("logs") / "baseline_v3_groupkfold"
SUMMARY_PATH = OUTPUT_DIR / "baseline_v3_summary.txt"
OOF_PREDICTIONS_PATH = OUTPUT_DIR / "baseline_v3_oof_predictions.csv"
FOLD_SCORES_PATH = OUTPUT_DIR / "baseline_v3_fold_scores.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "baseline_v3_confusion_matrix.png"
NORMALIZED_CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR / "baseline_v3_confusion_matrix_normalized.png"
)
PER_CLASS_METRICS_PATH = OUTPUT_DIR / "baseline_v3_per_class_metrics.png"

N_SPLITS = 5
KNN_PARAMS = {
    "n_neighbors": 5,
    "weights": "uniform",
    "metric": "minkowski",
    "p": 2,
}


def require_matplotlib():
    """导入 matplotlib；缺少依赖时给出清晰错误提示。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "缺少 matplotlib，无法生成 Baseline V3 图像。请先安装 matplotlib，"
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


def build_baseline_pipeline():
    """构建固定参数 KNN baseline Pipeline，避免标准化数据泄漏。"""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(**KNN_PARAMS)),
        ]
    )


def run_groupkfold_oof_evaluation(X, y, groups):
    """执行 GroupKFold OOF 评估，返回每个样本预测和每折指标。"""
    splitter = GroupKFold(n_splits=N_SPLITS)
    base_model = build_baseline_pipeline()
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
                "macro_f1": f1_score(
                    y_validation_fold,
                    validation_pred,
                    average="macro",
                    zero_division=0,
                ),
                "weighted_f1": f1_score(
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


def compute_metrics(y_true, y_pred, labels):
    """基于 OOF 预测计算总体指标、分类报告和混淆矩阵。"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        ),
        "classification_report_dict": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def save_confusion_matrix_image(matrix, labels, output_path):
    """保存显示原始数量的 OOF 混淆矩阵图像。"""
    plt = require_matplotlib()
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_title("Baseline V3 GroupKFold OOF Confusion Matrix")
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


def save_normalized_confusion_matrix_image(matrix, labels, output_path):
    """保存按真实类别数量归一化的 OOF 混淆矩阵图像。"""
    plt = require_matplotlib()
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

    axis.set_title("Baseline V3 GroupKFold OOF Normalized Confusion Matrix")
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


def save_per_class_metrics_image(report_dict, labels, output_path):
    """保存每个类别 precision、recall、f1-score 的柱状图。"""
    plt = require_matplotlib()
    per_class_metrics = pd.DataFrame(
        [
            {
                "label": label,
                "precision": report_dict[label]["precision"],
                "recall": report_dict[label]["recall"],
                "f1-score": report_dict[label]["f1-score"],
            }
            for label in labels
        ]
    )
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

    axis.set_title("Baseline V3 Per-Class Metrics")
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


def get_top_confusions(matrix, labels, top_n=10):
    """提取 OOF 混淆矩阵中误判数量最高的类别对。"""
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
    fold_scores,
    metrics,
    top_confusions,
):
    """生成 Baseline V3 GroupKFold 实验总结日志。"""
    matrix_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"true_{label}" for label in sorted(y.unique())],
        columns=[f"pred_{label}" for label in sorted(y.unique())],
    )

    summary_lines = [
        "音乐风格自动分类系统 - Baseline V3 GroupKFold",
        "=" * 60,
        "",
        "一、实验目的",
        "实现一个更严格、更公平的未调参 KNN baseline。",
        "该 baseline 使用 track_id 分组的 GroupKFold，用于后续调参模型的对照。",
        "",
        "二、数据路径",
        f"读取数据：{DATA_PATH.as_posix()}",
        "",
        "三、数据规模",
        f"样本数量：{len(y)}",
        f"track_id 数量：{groups.nunique()}",
        "",
        "四、类别分布",
        format_class_distribution(y),
        "",
        "五、字段处理",
        "目标变量：label",
        "分组字段：track_id",
        "删除的非模型字段：",
        ", ".join(dropped_columns),
        "",
        "六、模型设置",
        "使用的模型：KNN",
        f"KNN 固定参数：{KNN_PARAMS}",
        "说明：本 baseline 不进行调参，不使用 GridSearchCV，不搜索最优参数。",
        "",
        "七、验证方式",
        f"验证方式：GroupKFold(n_splits={N_SPLITS})",
        "分组字段：track_id",
        "",
        "八、Baseline V3 与 Baseline V2 的区别",
        "Baseline V3 与 Baseline V2 的主要区别在于评估方式。Baseline V2 使用片段级 train_test_split，可能存在同一首音乐的不同片段跨训练集和测试集的问题，因此结果可能偏乐观。Baseline V3 使用 GroupKFold，并以 track_id 为分组单位，能够保证同一首音乐的所有片段不会同时出现在训练折和验证折中，因此更适合作为后续调参实验的基准结果。",
        "",
        "九、5 折每折结果表",
        fold_scores.to_string(index=False),
        "",
        "十、OOF 总体指标",
        f"Accuracy：{metrics['accuracy']:.4f}",
        f"Macro Precision：{metrics['macro_precision']:.4f}",
        f"Macro Recall：{metrics['macro_recall']:.4f}",
        f"Macro F1：{metrics['macro_f1']:.4f}",
        f"Weighted F1：{metrics['weighted_f1']:.4f}",
        "",
        "十一、classification_report",
        metrics["classification_report"],
        "",
        "十二、confusion matrix 数值结果",
        matrix_df.to_string(),
        "",
        "十三、最容易混淆的前 10 个类别对",
        top_confusions.to_string(index=False) if not top_confusions.empty else "无",
        "",
        "十四、输出文件列表",
        SUMMARY_PATH.as_posix(),
        OOF_PREDICTIONS_PATH.as_posix(),
        FOLD_SCORES_PATH.as_posix(),
        CONFUSION_MATRIX_PATH.as_posix(),
        NORMALIZED_CONFUSION_MATRIX_PATH.as_posix(),
        PER_CLASS_METRICS_PATH.as_posix(),
        "",
        "实验完成。",
    ]

    return "\n".join(summary_lines)


def main():
    """执行 Baseline V3 GroupKFold OOF 评估，并保存日志、表格和图像。"""
    data_file = PROJECT_ROOT / DATA_PATH
    output_dir = PROJECT_ROOT / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file, encoding="utf-8")
    X, y, groups, dropped_columns = prepare_data(df)
    labels = sorted(y.unique())

    oof_predictions, fold_scores = run_groupkfold_oof_evaluation(X, y, groups)
    oof_predictions.to_csv(
        PROJECT_ROOT / OOF_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8",
    )
    fold_scores.to_csv(
        PROJECT_ROOT / FOLD_SCORES_PATH,
        index=False,
        encoding="utf-8",
    )

    metrics = compute_metrics(
        y_true=oof_predictions["y_true"],
        y_pred=oof_predictions["y_pred"],
        labels=labels,
    )
    save_confusion_matrix_image(
        metrics["confusion_matrix"],
        labels,
        PROJECT_ROOT / CONFUSION_MATRIX_PATH,
    )
    save_normalized_confusion_matrix_image(
        metrics["confusion_matrix"],
        labels,
        PROJECT_ROOT / NORMALIZED_CONFUSION_MATRIX_PATH,
    )
    save_per_class_metrics_image(
        metrics["classification_report_dict"],
        labels,
        PROJECT_ROOT / PER_CLASS_METRICS_PATH,
    )
    top_confusions = get_top_confusions(metrics["confusion_matrix"], labels)

    summary_log = build_summary_log(
        y=y,
        groups=groups,
        dropped_columns=dropped_columns,
        fold_scores=fold_scores,
        metrics=metrics,
        top_confusions=top_confusions,
    )
    (PROJECT_ROOT / SUMMARY_PATH).write_text(summary_log, encoding="utf-8")

    output_files = [
        SUMMARY_PATH,
        OOF_PREDICTIONS_PATH,
        FOLD_SCORES_PATH,
        CONFUSION_MATRIX_PATH,
        NORMALIZED_CONFUSION_MATRIX_PATH,
        PER_CLASS_METRICS_PATH,
    ]

    print("Baseline V3 GroupKFold 评估完成。输出文件如下：")
    for output_file in output_files:
        print(output_file.as_posix())


if __name__ == "__main__":
    main()
