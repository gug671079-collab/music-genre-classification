from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

import tune_knn_groupkfold as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path("data") / "processed" / "processed_features_3_sec_V2.csv"
SHARED_FINAL_TEST_IDS_PATH = Path("logs") / "final_test_track_ids.txt"
OUTPUT_DIR = Path("logs") / "knn_tuning_v2"
RESULTS_CSV_PATH = OUTPUT_DIR / "knn_tuning_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "knn_tuning_summary.txt"
TOP_PARAMS_IMAGE_PATH = OUTPUT_DIR / "knn_top_params.png"
K_CURVE_IMAGE_PATH = OUTPUT_DIR / "knn_k_curve.png"
HEATMAP_UNIFORM_IMAGE_PATH = OUTPUT_DIR / "knn_param_heatmap_uniform.png"
HEATMAP_DISTANCE_IMAGE_PATH = OUTPUT_DIR / "knn_param_heatmap_distance.png"
OOF_CONFUSION_MATRIX_IMAGE_PATH = OUTPUT_DIR / "knn_confusion_matrix.png"
OOF_NORMALIZED_CONFUSION_MATRIX_IMAGE_PATH = (
    OUTPUT_DIR / "knn_confusion_matrix_normalized.png"
)
PER_CLASS_METRICS_IMAGE_PATH = OUTPUT_DIR / "knn_per_class_metrics.png"
FOLD_SCORES_CSV_PATH = OUTPUT_DIR / "knn_fold_scores.csv"
FOLD_SCORES_IMAGE_PATH = OUTPUT_DIR / "knn_fold_scores.png"
OOF_PREDICTIONS_PATH = OUTPUT_DIR / "knn_oof_predictions.csv"
FINAL_TEST_PREDICTIONS_PATH = OUTPUT_DIR / "knn_final_test_predictions.csv"
FINAL_TEST_CONFUSION_MATRIX_IMAGE_PATH = OUTPUT_DIR / "knn_final_test_confusion_matrix.png"
FINAL_TEST_NORMALIZED_CONFUSION_MATRIX_IMAGE_PATH = (
    OUTPUT_DIR / "knn_final_test_confusion_matrix_normalized.png"
)

RANDOM_STATE = 42


def build_track_table(df):
    """构建 track_id 级别数据表，确保每首歌只对应一个 label。"""
    track_label_counts = df.groupby("track_id")["label"].nunique()
    inconsistent_tracks = track_label_counts[track_label_counts > 1]
    if not inconsistent_tracks.empty:
        raise ValueError(
            "存在同一个 track_id 对应多个 label 的情况："
            + ", ".join(inconsistent_tracks.index[:10])
        )

    return (
        df[["track_id", "label"]]
        .drop_duplicates(subset=["track_id"])
        .reset_index(drop=True)
    )


def get_or_create_final_test_track_ids(df):
    """读取或创建三个 V2 调参脚本共享的 final_test_track_ids。"""
    final_test_ids_file = PROJECT_ROOT / SHARED_FINAL_TEST_IDS_PATH
    final_test_ids_file.parent.mkdir(parents=True, exist_ok=True)

    if final_test_ids_file.exists():
        return [
            line.strip()
            for line in final_test_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    track_table = build_track_table(df)
    _, final_test_track_table = train_test_split(
        track_table,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=track_table["label"],
    )
    final_test_track_ids = sorted(final_test_track_table["track_id"].tolist())
    final_test_ids_file.write_text(
        "\n".join(final_test_track_ids) + "\n",
        encoding="utf-8",
    )

    return final_test_track_ids


def split_train_pool_and_final_test(df, final_test_track_ids):
    """按共享 final_test_track_ids 划分 train_pool 和 final_test。"""
    final_test_mask = df["track_id"].isin(final_test_track_ids)
    train_pool = df.loc[~final_test_mask].reset_index(drop=True)
    final_test = df.loc[final_test_mask].reset_index(drop=True)

    if train_pool.empty or final_test.empty:
        raise ValueError("train_pool 或 final_test 为空，请检查 final_test_track_ids。")

    return train_pool, final_test


def evaluate_final_test(best_params, X_train_pool, y_train_pool, X_final_test, final_test):
    """用最优参数在 train_pool 上训练，并只在 final_test 上做最终评估。"""
    model = base.build_knn_pipeline()
    model.set_params(**best_params)
    model.fit(X_train_pool, y_train_pool)
    y_pred = model.predict(X_final_test)

    return pd.DataFrame(
        {
            "sample_index": final_test.index,
            "track_id": final_test["track_id"],
            "y_true": final_test["label"],
            "y_pred": y_pred,
        }
    )


def save_confusion_matrix_image(y_true, y_pred, labels, output_path, title, normalized=False):
    """保存 final_test 混淆矩阵；normalized=True 时按真实类别行归一化。"""
    plt = base.require_matplotlib()
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

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


def build_summary_log(
    y_train_pool,
    groups_train_pool,
    final_test,
    final_test_track_ids,
    dropped_columns,
    grid_search,
    results_df,
    top_10_results,
    oof_predictions,
    fold_scores,
    per_class_metrics,
    final_test_predictions,
):
    """生成 KNN V2 调参日志，明确区分内部调参与外部测试。"""
    best_row = results_df.loc[grid_search.best_index_]
    oof_report = classification_report(
        oof_predictions["y_true"],
        oof_predictions["y_pred"],
        zero_division=0,
    )
    final_report = classification_report(
        final_test_predictions["y_true"],
        final_test_predictions["y_pred"],
        zero_division=0,
    )
    final_matrix = pd.DataFrame(
        confusion_matrix(
            final_test_predictions["y_true"],
            final_test_predictions["y_pred"],
            labels=sorted(y_train_pool.unique()),
        ),
        index=[f"true_{label}" for label in sorted(y_train_pool.unique())],
        columns=[f"pred_{label}" for label in sorted(y_train_pool.unique())],
    )

    return "\n".join(
        [
            "KNN GroupKFold 调参实验 V2",
            "=" * 50,
            "",
            "一、实验设置",
            f"读取数据：{DATA_PATH.as_posix()}",
            f"共享 final_test_track_ids：{SHARED_FINAL_TEST_IDS_PATH.as_posix()}",
            "本实验先以 track_id 为单位固定 20% final_test。",
            "GridSearchCV 只在剩余 80% train_pool 内部进行。",
            "final_test 不参与参数选择，只用于最终评估。",
            "因此 final_test 分数可以用于三个模型之间的最终横向比较。",
            "",
            "二、数据规模",
            f"train_pool 样本数量：{len(y_train_pool)}",
            f"train_pool track_id 数量：{groups_train_pool.nunique()}",
            f"final_test 样本数量：{len(final_test)}",
            f"final_test track_id 数量：{len(final_test_track_ids)}",
            "",
            "三、字段处理",
            "目标变量：label",
            "分组字段：track_id",
            "删除的非模型字段：",
            ", ".join(dropped_columns),
            "",
            "四、内部调参结果",
            f"验证方式：GroupKFold(n_splits={base.N_SPLITS})",
            "主要评价指标：Macro-F1",
            f"参数搜索范围：{base.get_param_grid()}",
            f"最佳参数：{grid_search.best_params_}",
            f"best mean_test_f1_macro：{best_row['mean_test_f1_macro']:.4f}",
            f"std_test_f1_macro：{best_row['std_test_f1_macro']:.4f}",
            f"mean_test_accuracy：{best_row['mean_test_accuracy']:.4f}",
            f"mean_test_f1_weighted：{best_row['mean_test_f1_weighted']:.4f}",
            "",
            "五、排名前 10 的参数组合",
            top_10_results.to_string(index=False),
            "",
            "六、train_pool 内部 OOF 结果",
            f"OOF Accuracy：{accuracy_score(oof_predictions['y_true'], oof_predictions['y_pred']):.4f}",
            f"OOF Macro F1：{f1_score(oof_predictions['y_true'], oof_predictions['y_pred'], average='macro', zero_division=0):.4f}",
            "OOF classification_report：",
            oof_report,
            "OOF 每类指标：",
            per_class_metrics.to_string(index=False),
            "OOF 每折结果：",
            fold_scores.to_string(index=False),
            "",
            "七、final_test 外部测试结果",
            f"external_test_accuracy：{accuracy_score(final_test_predictions['y_true'], final_test_predictions['y_pred']):.4f}",
            f"external_test_macro_f1：{f1_score(final_test_predictions['y_true'], final_test_predictions['y_pred'], average='macro', zero_division=0):.4f}",
            "final_test classification_report：",
            final_report,
            "final_test confusion_matrix：",
            final_matrix.to_string(),
            "",
            "八、输出文件列表",
            SHARED_FINAL_TEST_IDS_PATH.as_posix(),
            RESULTS_CSV_PATH.as_posix(),
            SUMMARY_PATH.as_posix(),
            TOP_PARAMS_IMAGE_PATH.as_posix(),
            K_CURVE_IMAGE_PATH.as_posix(),
            HEATMAP_UNIFORM_IMAGE_PATH.as_posix(),
            HEATMAP_DISTANCE_IMAGE_PATH.as_posix(),
            OOF_CONFUSION_MATRIX_IMAGE_PATH.as_posix(),
            OOF_NORMALIZED_CONFUSION_MATRIX_IMAGE_PATH.as_posix(),
            PER_CLASS_METRICS_IMAGE_PATH.as_posix(),
            FOLD_SCORES_CSV_PATH.as_posix(),
            FOLD_SCORES_IMAGE_PATH.as_posix(),
            OOF_PREDICTIONS_PATH.as_posix(),
            FINAL_TEST_PREDICTIONS_PATH.as_posix(),
            FINAL_TEST_CONFUSION_MATRIX_IMAGE_PATH.as_posix(),
            FINAL_TEST_NORMALIZED_CONFUSION_MATRIX_IMAGE_PATH.as_posix(),
            "",
            "实验完成。",
        ]
    )


def main():
    """执行 KNN V2：train_pool 内部调参 + final_test 外部测试。"""
    output_dir = PROJECT_ROOT / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROJECT_ROOT / DATA_PATH, encoding="utf-8")
    final_test_track_ids = get_or_create_final_test_track_ids(df)
    train_pool, final_test = split_train_pool_and_final_test(df, final_test_track_ids)

    X_train_pool, y_train_pool, groups_train_pool, dropped_columns = base.prepare_data(
        train_pool
    )
    X_final_test, _, _, _ = base.prepare_data(final_test)

    grid_search = base.run_grid_search(X_train_pool, y_train_pool, groups_train_pool)
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(PROJECT_ROOT / RESULTS_CSV_PATH, index=False, encoding="utf-8")

    top_10_results = base.get_top_10_results(results_df)
    base.plot_top_params(top_10_results, PROJECT_ROOT / TOP_PARAMS_IMAGE_PATH)
    base.plot_k_curve(results_df, PROJECT_ROOT / K_CURVE_IMAGE_PATH)
    base.plot_param_heatmap(
        results_df,
        weights="uniform",
        output_path=PROJECT_ROOT / HEATMAP_UNIFORM_IMAGE_PATH,
    )
    base.plot_param_heatmap(
        results_df,
        weights="distance",
        output_path=PROJECT_ROOT / HEATMAP_DISTANCE_IMAGE_PATH,
    )

    oof_predictions, fold_scores = base.generate_oof_predictions_and_fold_scores(
        X_train_pool,
        y_train_pool,
        groups_train_pool,
        grid_search.best_params_,
    )
    oof_predictions.to_csv(
        PROJECT_ROOT / OOF_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    fold_scores.to_csv(
        PROJECT_ROOT / FOLD_SCORES_CSV_PATH, index=False, encoding="utf-8"
    )

    labels = sorted(y_train_pool.unique())
    base.plot_confusion_matrix(
        oof_predictions["y_true"],
        oof_predictions["y_pred"],
        labels,
        PROJECT_ROOT / OOF_CONFUSION_MATRIX_IMAGE_PATH,
    )
    base.plot_normalized_confusion_matrix(
        oof_predictions["y_true"],
        oof_predictions["y_pred"],
        labels,
        PROJECT_ROOT / OOF_NORMALIZED_CONFUSION_MATRIX_IMAGE_PATH,
    )
    per_class_metrics = base.compute_per_class_metrics(
        oof_predictions["y_true"],
        oof_predictions["y_pred"],
        labels,
    )
    base.plot_per_class_metrics(
        per_class_metrics,
        PROJECT_ROOT / PER_CLASS_METRICS_IMAGE_PATH,
    )
    base.plot_fold_scores(fold_scores, PROJECT_ROOT / FOLD_SCORES_IMAGE_PATH)

    final_test_predictions = evaluate_final_test(
        grid_search.best_params_,
        X_train_pool,
        y_train_pool,
        X_final_test,
        final_test,
    )
    final_test_predictions.to_csv(
        PROJECT_ROOT / FINAL_TEST_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8",
    )
    save_confusion_matrix_image(
        final_test_predictions["y_true"],
        final_test_predictions["y_pred"],
        labels,
        PROJECT_ROOT / FINAL_TEST_CONFUSION_MATRIX_IMAGE_PATH,
        "KNN V2 Final Test Confusion Matrix",
    )
    save_confusion_matrix_image(
        final_test_predictions["y_true"],
        final_test_predictions["y_pred"],
        labels,
        PROJECT_ROOT / FINAL_TEST_NORMALIZED_CONFUSION_MATRIX_IMAGE_PATH,
        "KNN V2 Final Test Normalized Confusion Matrix",
        normalized=True,
    )

    summary_log = build_summary_log(
        y_train_pool,
        groups_train_pool,
        final_test,
        final_test_track_ids,
        dropped_columns,
        grid_search,
        results_df,
        top_10_results,
        oof_predictions,
        fold_scores,
        per_class_metrics,
        final_test_predictions,
    )
    (PROJECT_ROOT / SUMMARY_PATH).write_text(summary_log, encoding="utf-8")

    output_files = [
        SHARED_FINAL_TEST_IDS_PATH,
        RESULTS_CSV_PATH,
        SUMMARY_PATH,
        TOP_PARAMS_IMAGE_PATH,
        K_CURVE_IMAGE_PATH,
        HEATMAP_UNIFORM_IMAGE_PATH,
        HEATMAP_DISTANCE_IMAGE_PATH,
        OOF_CONFUSION_MATRIX_IMAGE_PATH,
        OOF_NORMALIZED_CONFUSION_MATRIX_IMAGE_PATH,
        PER_CLASS_METRICS_IMAGE_PATH,
        FOLD_SCORES_CSV_PATH,
        FOLD_SCORES_IMAGE_PATH,
        OOF_PREDICTIONS_PATH,
        FINAL_TEST_PREDICTIONS_PATH,
        FINAL_TEST_CONFUSION_MATRIX_IMAGE_PATH,
        FINAL_TEST_NORMALIZED_CONFUSION_MATRIX_IMAGE_PATH,
    ]

    print("KNN GroupKFold V2 调参实验完成。输出文件如下：")
    for output_file in output_files:
        print(output_file.as_posix())


if __name__ == "__main__":
    main()
