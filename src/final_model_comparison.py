from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ===== 路径设置 =====
# 本文件位于 src 目录下，parents[1] 可以定位到项目根目录。
# 本脚本只读取已有 V2 实验输出，不重新训练任何模型。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path("logs") / "final_model_comparison"

MODEL_CONFIGS = {
    "KNN": {
        "prediction_path": Path("logs")
        / "knn_tuning_v2"
        / "knn_final_test_predictions.csv",
        "tuning_results_path": Path("logs")
        / "knn_tuning_v2"
        / "knn_tuning_results.csv",
    },
    "SVM-RBF": {
        "prediction_path": Path("logs")
        / "svm_rbf_tuning_v2"
        / "svm_rbf_final_test_predictions.csv",
        "tuning_results_path": Path("logs")
        / "svm_rbf_tuning_v2"
        / "svm_rbf_tuning_results.csv",
    },
    "RandomForest": {
        "prediction_path": Path("logs")
        / "random_forest_tuning_v2"
        / "random_forest_final_test_predictions.csv",
        "tuning_results_path": Path("logs")
        / "random_forest_tuning_v2"
        / "random_forest_tuning_results.csv",
    },
}

RESULTS_CSV_PATH = OUTPUT_DIR / "final_model_comparison_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "final_model_comparison_summary.txt"
BARPLOT_PATH = OUTPUT_DIR / "final_model_comparison_barplot.png"
BARPLOT_ZOOMED_PATH = OUTPUT_DIR / "final_model_comparison_barplot_zoomed.png"
INTERNAL_EXTERNAL_F1_PATH = OUTPUT_DIR / "internal_vs_external_macro_f1.png"
INTERNAL_EXTERNAL_F1_ZOOMED_PATH = (
    OUTPUT_DIR / "internal_vs_external_macro_f1_zoomed.png"
)
PER_CLASS_RECALL_PATH = OUTPUT_DIR / "per_class_recall_comparison.png"
PER_CLASS_F1_PATH = OUTPUT_DIR / "per_class_f1_comparison.png"
BEST_MODEL_CM_PATH = OUTPUT_DIR / "best_model_confusion_matrix.png"
BEST_MODEL_NORMALIZED_CM_PATH = OUTPUT_DIR / "best_model_confusion_matrix_normalized.png"


def require_matplotlib():
    """导入 matplotlib；缺少依赖时给出清晰错误提示。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "缺少 matplotlib，无法生成最终模型对比图像。请先安装 matplotlib，"
            "或在包含 matplotlib 的 Python 环境中运行本脚本。"
        ) from error

    return plt


def read_required_csv(path):
    """读取必要 CSV 文件；不存在时给出清晰错误提示。"""
    file_path = PROJECT_ROOT / path
    if not file_path.exists():
        raise FileNotFoundError(f"缺少必要输入文件：{path.as_posix()}")

    return pd.read_csv(file_path, encoding="utf-8")


def validate_prediction_columns(model_name, predictions):
    """检查 final_test prediction 文件中的必要字段。"""
    required_columns = ["sample_index", "track_id", "y_true", "y_pred"]
    missing_columns = [
        column for column in required_columns if column not in predictions.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{model_name} 的 final_test_predictions 缺少字段："
            + ", ".join(missing_columns)
        )


def load_inputs():
    """读取三个模型的 final_test 预测和调参结果。"""
    inputs = {}

    for model_name, config in MODEL_CONFIGS.items():
        predictions = read_required_csv(config["prediction_path"])
        tuning_results = read_required_csv(config["tuning_results_path"])
        validate_prediction_columns(model_name, predictions)

        inputs[model_name] = {
            "predictions": predictions,
            "tuning_results": tuning_results,
        }

    return inputs


def validate_same_final_test(inputs):
    """检查三个模型是否使用完全相同的 final_test 样本。"""
    reference_model = next(iter(inputs))
    reference = inputs[reference_model]["predictions"][
        ["sample_index", "track_id", "y_true"]
    ].reset_index(drop=True)

    for model_name, data in inputs.items():
        current = data["predictions"][
            ["sample_index", "track_id", "y_true"]
        ].reset_index(drop=True)
        if not reference.equals(current):
            raise ValueError(
                f"{model_name} 的 final_test 与 {reference_model} 不一致，"
                "不能进行公平横向比较。"
            )


def get_best_tuning_row(tuning_results):
    """读取 rank_test_f1_macro 排名第一的参数组合。"""
    if "rank_test_f1_macro" not in tuning_results.columns:
        raise ValueError("调参结果缺少 rank_test_f1_macro 字段。")

    return tuning_results.sort_values("rank_test_f1_macro").iloc[0]


def extract_best_params(best_row):
    """从 GridSearchCV 结果行中提取 param_ 开头的最佳参数。"""
    param_items = {
        column.replace("param_", ""): best_row[column]
        for column in best_row.index
        if column.startswith("param_")
    }

    return str(param_items)


def compute_model_metrics(model_name, predictions, tuning_results, labels):
    """统一计算单个模型的外部测试指标和调参指标。"""
    best_row = get_best_tuning_row(tuning_results)
    report_dict = classification_report(
        predictions["y_true"],
        predictions["y_pred"],
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        predictions["y_true"],
        predictions["y_pred"],
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(
        predictions["y_true"],
        predictions["y_pred"],
        labels=labels,
    )

    per_class_rows = []
    for label in labels:
        per_class_rows.append(
            {
                "model": model_name,
                "label": label,
                "precision": report_dict[label]["precision"],
                "recall": report_dict[label]["recall"],
                "f1-score": report_dict[label]["f1-score"],
                "support": int(report_dict[label]["support"]),
            }
        )

    return {
        "summary_row": {
            "model": model_name,
            "best_params": extract_best_params(best_row),
            "internal_cv_macro_f1_mean": best_row["mean_test_f1_macro"],
            "internal_cv_macro_f1_std": best_row["std_test_f1_macro"],
            "internal_cv_accuracy_mean": best_row["mean_test_accuracy"],
            "internal_cv_weighted_f1_mean": best_row["mean_test_f1_weighted"],
            "external_test_accuracy": accuracy_score(
                predictions["y_true"],
                predictions["y_pred"],
            ),
            "external_test_macro_f1": f1_score(
                predictions["y_true"],
                predictions["y_pred"],
                average="macro",
                zero_division=0,
            ),
        },
        "classification_report": report_text,
        "confusion_matrix": matrix,
        "per_class_metrics": pd.DataFrame(per_class_rows),
    }


def compute_all_metrics(inputs):
    """计算三个模型的所有对比指标。"""
    labels = sorted(next(iter(inputs.values()))["predictions"]["y_true"].unique())
    model_metrics = {}
    summary_rows = []
    per_class_tables = []

    for model_name, data in inputs.items():
        metrics = compute_model_metrics(
            model_name,
            data["predictions"],
            data["tuning_results"],
            labels,
        )
        model_metrics[model_name] = metrics
        summary_rows.append(metrics["summary_row"])
        per_class_tables.append(metrics["per_class_metrics"])

    return (
        pd.DataFrame(summary_rows),
        pd.concat(per_class_tables, ignore_index=True),
        model_metrics,
        labels,
    )


def select_best_model(results_df):
    """默认按 external_test_macro_f1 选择最佳模型，必要时参考 accuracy。"""
    return (
        results_df.sort_values(
            ["external_test_macro_f1", "external_test_accuracy"],
            ascending=False,
        )
        .iloc[0]["model"]
    )


def add_bar_labels(axis, bars):
    """给柱状图添加数值标签。"""
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.003,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def apply_zoomed_axis(axis, values):
    """为放大差异版柱状图自动设置纵轴，并添加说明文字。"""
    y_min = max(0, min(values) - 0.03)
    y_max = min(1.0, max(values) + 0.03)

    if y_max - y_min < 0.06:
        center = (y_max + y_min) / 2
        y_min = max(0, center - 0.03)
        y_max = min(1.0, center + 0.03)

    axis.set_ylim(y_min, y_max)
    axis.text(
        0.5,
        0.96,
        "Note: the y-axis is truncated to make small differences easier to compare.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="dimgray",
    )


def plot_overall_metrics(results_df, output_path, zoomed=False):
    """绘制 external_test_accuracy 与 external_test_macro_f1 对比图。"""
    plt = require_matplotlib()
    models = results_df["model"].tolist()
    x_positions = np.arange(len(models))
    width = 0.35

    figure, axis = plt.subplots(figsize=(9, 6))
    accuracy_bars = axis.bar(
        x_positions - width / 2,
        results_df["external_test_accuracy"],
        width=width,
        label="External Test Accuracy",
    )
    macro_f1_bars = axis.bar(
        x_positions + width / 2,
        results_df["external_test_macro_f1"],
        width=width,
        label="External Test Macro F1",
    )

    title = "Final Model Comparison on Shared Final Test"
    if zoomed:
        title += " (Zoomed)"
    axis.set_title(title)
    axis.set_xlabel("Model")
    axis.set_ylabel("Score")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(models)
    values = [
        *results_df["external_test_accuracy"].tolist(),
        *results_df["external_test_macro_f1"].tolist(),
    ]
    if zoomed:
        apply_zoomed_axis(axis, values)
    else:
        axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    add_bar_labels(axis, accuracy_bars)
    add_bar_labels(axis, macro_f1_bars)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_internal_vs_external_f1(results_df, output_path, zoomed=False):
    """绘制内部验证 Macro F1 与外部测试 Macro F1 对比图。"""
    plt = require_matplotlib()
    models = results_df["model"].tolist()
    x_positions = np.arange(len(models))
    width = 0.35

    figure, axis = plt.subplots(figsize=(9, 6))
    internal_bars = axis.bar(
        x_positions - width / 2,
        results_df["internal_cv_macro_f1_mean"],
        width=width,
        label="Internal CV Macro F1",
    )
    external_bars = axis.bar(
        x_positions + width / 2,
        results_df["external_test_macro_f1"],
        width=width,
        label="External Test Macro F1",
    )

    title = "Internal CV Macro F1 vs External Test Macro F1"
    if zoomed:
        title += " (Zoomed)"
    axis.set_title(title)
    axis.set_xlabel("Model")
    axis.set_ylabel("Macro F1")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(models)
    values = [
        *results_df["internal_cv_macro_f1_mean"].tolist(),
        *results_df["external_test_macro_f1"].tolist(),
    ]
    if zoomed:
        apply_zoomed_axis(axis, values)
    else:
        axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    add_bar_labels(axis, internal_bars)
    add_bar_labels(axis, external_bars)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_per_class_metric(per_class_df, metric_column, title, output_path):
    """绘制每类别 recall 或 f1-score 的模型对比图。"""
    plt = require_matplotlib()
    labels = per_class_df["label"].drop_duplicates().tolist()
    models = per_class_df["model"].drop_duplicates().tolist()
    x_positions = np.arange(len(labels))
    width = 0.24

    figure, axis = plt.subplots(figsize=(14, 6))
    for model_index, model_name in enumerate(models):
        model_values = (
            per_class_df.loc[per_class_df["model"] == model_name]
            .set_index("label")
            .loc[labels, metric_column]
        )
        offsets = x_positions + (model_index - 1) * width
        axis.bar(offsets, model_values, width=width, label=model_name)

    axis.set_title(title)
    axis.set_xlabel("Music Genre")
    axis.set_ylabel(metric_column)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.4)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(matrix, labels, output_path, title, normalized=False):
    """绘制最佳模型混淆矩阵。"""
    plt = require_matplotlib()

    if normalized:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_to_show = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0,
        )
    else:
        matrix_to_show = matrix

    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix_to_show, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)

    for row_index in range(matrix_to_show.shape[0]):
        for column_index in range(matrix_to_show.shape[1]):
            text = (
                f"{matrix_to_show[row_index, column_index]:.2f}"
                if normalized
                else str(matrix_to_show[row_index, column_index])
            )
            axis.text(column_index, row_index, text, ha="center", va="center")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_summary_log(results_df, model_metrics, best_model):
    """生成最终模型对比文字日志。"""
    lines = [
        "最终模型横向对比",
        "=" * 50,
        "",
        "一、实验说明",
        "三个模型均基于同一个 final_test 进行比较。",
        "本脚本不重新训练模型，只读取三个 V2 调参脚本已经生成的输出结果。",
        "最终模型选择主要依据 external_test_macro_f1，其次参考 accuracy 和各类别表现。",
        "",
        "二、总体对比表",
        results_df.to_string(index=False),
        "",
        "三、最佳模型判断",
        f"最佳模型：{best_model}",
        f"选择依据：{best_model} 的 external_test_macro_f1 最高；如分数接近，再参考 external_test_accuracy 和每类别表现是否均衡。",
        "",
        "四、各模型最佳参数与核心指标",
    ]

    for _, row in results_df.iterrows():
        lines.extend(
            [
                "",
                f"模型：{row['model']}",
                f"最佳参数：{row['best_params']}",
                f"internal_cv_macro_f1_mean：{row['internal_cv_macro_f1_mean']:.4f}",
                f"external_test_accuracy：{row['external_test_accuracy']:.4f}",
                f"external_test_macro_f1：{row['external_test_macro_f1']:.4f}",
            ]
        )

    lines.append("")
    lines.append("五、classification_report")
    for model_name, metrics in model_metrics.items():
        lines.extend(
            [
                "",
                f"模型：{model_name}",
                metrics["classification_report"],
            ]
        )

    lines.extend(
        [
            "",
            "六、输出文件列表",
            RESULTS_CSV_PATH.as_posix(),
            SUMMARY_PATH.as_posix(),
            BARPLOT_PATH.as_posix(),
            INTERNAL_EXTERNAL_F1_PATH.as_posix(),
            PER_CLASS_RECALL_PATH.as_posix(),
            PER_CLASS_F1_PATH.as_posix(),
            BEST_MODEL_CM_PATH.as_posix(),
            BEST_MODEL_NORMALIZED_CM_PATH.as_posix(),
            "",
            "对比完成。",
        ]
    )

    return "\n".join(lines)


def main():
    """读取三个 V2 模型输出，生成最终横向对比结果。"""
    output_dir = PROJECT_ROOT / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    validate_same_final_test(inputs)
    results_df, per_class_df, model_metrics, labels = compute_all_metrics(inputs)
    best_model = select_best_model(results_df)

    results_df.to_csv(PROJECT_ROOT / RESULTS_CSV_PATH, index=False, encoding="utf-8")

    plot_overall_metrics(results_df, PROJECT_ROOT / BARPLOT_PATH)
    plot_overall_metrics(
        results_df,
        PROJECT_ROOT / BARPLOT_ZOOMED_PATH,
        zoomed=True,
    )
    plot_internal_vs_external_f1(results_df, PROJECT_ROOT / INTERNAL_EXTERNAL_F1_PATH)
    plot_internal_vs_external_f1(
        results_df,
        PROJECT_ROOT / INTERNAL_EXTERNAL_F1_ZOOMED_PATH,
        zoomed=True,
    )
    plot_per_class_metric(
        per_class_df,
        metric_column="recall",
        title="Per-class Recall Comparison",
        output_path=PROJECT_ROOT / PER_CLASS_RECALL_PATH,
    )
    plot_per_class_metric(
        per_class_df,
        metric_column="f1-score",
        title="Per-class F1-score Comparison",
        output_path=PROJECT_ROOT / PER_CLASS_F1_PATH,
    )
    plot_confusion_matrix(
        model_metrics[best_model]["confusion_matrix"],
        labels,
        PROJECT_ROOT / BEST_MODEL_CM_PATH,
        f"Best Model Confusion Matrix - {best_model}",
    )
    plot_confusion_matrix(
        model_metrics[best_model]["confusion_matrix"],
        labels,
        PROJECT_ROOT / BEST_MODEL_NORMALIZED_CM_PATH,
        f"Best Model Normalized Confusion Matrix - {best_model}",
        normalized=True,
    )

    summary_log = build_summary_log(results_df, model_metrics, best_model)
    (PROJECT_ROOT / SUMMARY_PATH).write_text(summary_log, encoding="utf-8")

    output_files = [
        RESULTS_CSV_PATH,
        SUMMARY_PATH,
        BARPLOT_PATH,
        BARPLOT_ZOOMED_PATH,
        INTERNAL_EXTERNAL_F1_PATH,
        INTERNAL_EXTERNAL_F1_ZOOMED_PATH,
        PER_CLASS_RECALL_PATH,
        PER_CLASS_F1_PATH,
        BEST_MODEL_CM_PATH,
        BEST_MODEL_NORMALIZED_CM_PATH,
    ]

    print("最终模型横向对比完成。输出文件如下：")
    for output_file in output_files:
        print(output_file.as_posix())


if __name__ == "__main__":
    main()
