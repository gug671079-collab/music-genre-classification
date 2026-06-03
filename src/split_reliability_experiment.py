from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ===== 路径设置 =====
# 本文件位于 src 目录下，parents[1] 可以定位到项目根目录。
# 下面只使用项目内的相对路径，不写死任何本地绝对路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path("data") / "processed" / "processed_features_3_sec_V2.csv"
LOG_DIR = Path("logs") / "split_reliability_experiment"
SUMMARY_PATH = LOG_DIR / "split_reliability_summary.txt"
FINAL_TEST_TRACK_IDS_PATH = LOG_DIR / "final_test_track_ids.txt"
RESULTS_CSV_PATH = LOG_DIR / "split_reliability_results.csv"
INTERNAL_EXTERNAL_F1_IMAGE_PATH = (
    LOG_DIR / "split_reliability_internal_vs_external_f1.png"
)
OPTIMISM_GAP_IMAGE_PATH = LOG_DIR / "split_reliability_optimism_gap.png"
TRACK_OVERLAP_IMAGE_PATH = LOG_DIR / "split_reliability_track_overlap.png"
EXTERNAL_F1_IMAGE_PATH = LOG_DIR / "split_reliability_external_f1.png"

RANDOM_STATE = 42


def make_pipeline(model):
    """把 StandardScaler 和模型组合成统一训练流程。"""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def get_models():
    """定义本实验参与比较的三个基础模型。"""
    return {
        "KNN": make_pipeline(
            KNeighborsClassifier(
                n_neighbors=5,
                weights="uniform",
                metric="minkowski",
                p=2,
            )
        ),
        "SVM_RBF": make_pipeline(
            SVC(
                kernel="rbf",
                C=1,
                gamma="scale",
            )
        ),
        "RandomForest": make_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
            )
        ),
    }


def prepare_data(df):
    """检查必要字段，并整理 X、y、track_id。"""
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


def build_track_table(df):
    """构建 track_id 级别的数据表，每个 track_id 只保留一条 label。"""
    track_label_counts = df.groupby("track_id")["label"].nunique()
    inconsistent_tracks = track_label_counts[track_label_counts > 1]
    if not inconsistent_tracks.empty:
        raise ValueError(
            "存在同一个 track_id 对应多个 label 的情况，请先检查数据："
            + ", ".join(inconsistent_tracks.index[:10])
        )

    return (
        df[["track_id", "label"]]
        .drop_duplicates(subset=["track_id"])
        .reset_index(drop=True)
    )


def split_final_test(df):
    """在 track_id 层面划分固定最终外部测试集。"""
    track_table = build_track_table(df)
    train_track_table, final_test_track_table = train_test_split(
        track_table,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=track_table["label"],
    )

    final_test_track_ids = set(final_test_track_table["track_id"])
    final_test_mask = df["track_id"].isin(final_test_track_ids)
    final_test = df.loc[final_test_mask].reset_index(drop=True)
    train_pool = df.loc[~final_test_mask].reset_index(drop=True)

    return train_pool, final_test, sorted(final_test_track_ids)


def calculate_track_overlap_ratio(train_groups, validation_groups):
    """计算验证折样本中 track_id 也出现在训练折中的比例。"""
    train_track_ids = set(train_groups)
    return validation_groups.isin(train_track_ids).mean()


def evaluate_internal_cv(model, X, y, groups, validation_method):
    """计算一种内部验证方式下的 5 折 Macro F1 和 track_id 重叠比例。"""
    fold_macro_f1_scores = []
    fold_track_overlap_ratios = []

    if validation_method == "StratifiedKFold":
        splitter = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        split_iterator = splitter.split(X, y)
    elif validation_method == "GroupKFold":
        splitter = GroupKFold(n_splits=5)
        split_iterator = splitter.split(X, y, groups=groups)
    else:
        raise ValueError(f"未知验证方式：{validation_method}")

    for train_index, validation_index in split_iterator:
        X_train_fold = X.iloc[train_index]
        X_validation_fold = X.iloc[validation_index]
        y_train_fold = y.iloc[train_index]
        y_validation_fold = y.iloc[validation_index]
        train_groups = groups.iloc[train_index]
        validation_groups = groups.iloc[validation_index]

        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold)
        y_validation_pred = fold_model.predict(X_validation_fold)

        fold_macro_f1_scores.append(
            f1_score(
                y_validation_fold,
                y_validation_pred,
                average="macro",
                zero_division=0,
            )
        )
        fold_track_overlap_ratios.append(
            calculate_track_overlap_ratio(train_groups, validation_groups)
        )

    macro_f1_series = pd.Series(fold_macro_f1_scores)
    overlap_series = pd.Series(fold_track_overlap_ratios)

    return {
        "internal_cv_macro_f1_mean": macro_f1_series.mean(),
        "internal_cv_macro_f1_std": macro_f1_series.std(ddof=1),
        "cv_track_id_overlap_ratio_mean": overlap_series.mean(),
        "cv_track_id_overlap_ratio_std": overlap_series.std(ddof=1),
    }


def evaluate_external_test(model, X_train_pool, y_train_pool, X_final_test, y_final_test):
    """用整个 train_pool 训练模型，并在固定 final_test 上评估。"""
    final_model = clone(model)
    final_model.fit(X_train_pool, y_train_pool)
    y_pred = final_model.predict(X_final_test)

    return {
        "external_test_accuracy": accuracy_score(y_final_test, y_pred),
        "external_test_macro_f1": f1_score(
            y_final_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def run_experiment(train_pool, final_test):
    """对每个模型、每种验证方式运行可靠性实验。"""
    X_train_pool, y_train_pool, train_groups, dropped_columns = prepare_data(train_pool)
    X_final_test, y_final_test, _, _ = prepare_data(final_test)

    models = get_models()
    validation_methods = ["StratifiedKFold", "GroupKFold"]
    results = []

    for model_name, model in models.items():
        for validation_method in validation_methods:
            internal_metrics = evaluate_internal_cv(
                model=model,
                X=X_train_pool,
                y=y_train_pool,
                groups=train_groups,
                validation_method=validation_method,
            )
            external_metrics = evaluate_external_test(
                model=model,
                X_train_pool=X_train_pool,
                y_train_pool=y_train_pool,
                X_final_test=X_final_test,
                y_final_test=y_final_test,
            )
            optimism_gap = (
                internal_metrics["internal_cv_macro_f1_mean"]
                - external_metrics["external_test_macro_f1"]
            )

            results.append(
                {
                    "model": model_name,
                    "validation_method": validation_method,
                    **internal_metrics,
                    **external_metrics,
                    "optimism_gap": optimism_gap,
                }
            )

    return pd.DataFrame(results), dropped_columns


def require_matplotlib():
    """导入 matplotlib；如果缺少依赖，给出清晰错误提示。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "缺少 matplotlib，无法生成实验图像。请先安装 matplotlib，"
            "或在包含 matplotlib 的 Python 环境中运行本脚本。"
        ) from error

    return plt


def plot_internal_vs_external_f1(results_df, output_path):
    """绘制内部验证 Macro F1 与外部测试 Macro F1 对比图。"""
    plt = require_matplotlib()
    figure, axis = plt.subplots(figsize=(12, 7))

    labels = [
        f"{row.model}\n{row.validation_method}"
        for row in results_df.itertuples(index=False)
    ]
    x_positions = range(len(labels))
    width = 0.35

    axis.bar(
        [position - width / 2 for position in x_positions],
        results_df["internal_cv_macro_f1_mean"],
        width=width,
        label="Internal CV Macro F1",
    )
    axis.bar(
        [position + width / 2 for position in x_positions],
        results_df["external_test_macro_f1"],
        width=width,
        label="External Test Macro F1",
    )

    axis.set_title("Internal CV Macro F1 vs External Test Macro F1")
    axis.set_xlabel("Model and Validation Method")
    axis.set_ylabel("Macro F1")
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.4)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_grouped_bar(results_df, metric_column, title, ylabel, output_path):
    """绘制按模型和验证方式分组的柱状图。"""
    plt = require_matplotlib()
    models = list(results_df["model"].drop_duplicates())
    validation_methods = list(results_df["validation_method"].drop_duplicates())
    x_positions = range(len(models))
    width = 0.35

    figure, axis = plt.subplots(figsize=(10, 6))

    for method_index, validation_method in enumerate(validation_methods):
        method_values = []
        for model_name in models:
            value = results_df.loc[
                (results_df["model"] == model_name)
                & (results_df["validation_method"] == validation_method),
                metric_column,
            ].iloc[0]
            method_values.append(value)

        offsets = [
            position + (method_index - 0.5) * width
            for position in x_positions
        ]
        axis.bar(
            offsets,
            method_values,
            width=width,
            label=validation_method,
        )

    axis.set_title(title)
    axis.set_xlabel("Model")
    axis.set_ylabel(ylabel)
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(models)
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.4)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_plots(results_df):
    """保存本实验要求的四张对比图。"""
    output_files = [
        INTERNAL_EXTERNAL_F1_IMAGE_PATH,
        OPTIMISM_GAP_IMAGE_PATH,
        TRACK_OVERLAP_IMAGE_PATH,
        EXTERNAL_F1_IMAGE_PATH,
    ]

    plot_internal_vs_external_f1(
        results_df,
        PROJECT_ROOT / INTERNAL_EXTERNAL_F1_IMAGE_PATH,
    )
    plot_grouped_bar(
        results_df,
        metric_column="optimism_gap",
        title="Optimism Gap by Validation Method",
        ylabel="Optimism Gap",
        output_path=PROJECT_ROOT / OPTIMISM_GAP_IMAGE_PATH,
    )
    plot_grouped_bar(
        results_df,
        metric_column="cv_track_id_overlap_ratio_mean",
        title="Track ID Overlap Ratio by Validation Method",
        ylabel="Track ID Overlap Ratio",
        output_path=PROJECT_ROOT / TRACK_OVERLAP_IMAGE_PATH,
    )
    plot_grouped_bar(
        results_df,
        metric_column="external_test_macro_f1",
        title="External Test Macro F1 by Validation Method",
        ylabel="External Test Macro F1",
        output_path=PROJECT_ROOT / EXTERNAL_F1_IMAGE_PATH,
    )

    return output_files


def format_label_distribution(df):
    """整理类别分布文本。"""
    return df["label"].value_counts().sort_index().to_string()


def build_summary_log(
    results_df,
    dropped_columns,
    train_pool,
    final_test,
    final_test_track_ids,
):
    """生成文字版实验总结日志。"""
    summary_lines = [
        "音乐风格自动分类系统 - 划分方式可靠性实验",
        "=" * 60,
        "",
        "一、实验目的",
        "比较普通随机片段验证 StratifiedKFold 与按 track_id 分组验证 GroupKFold 的可靠性。",
        "重点观察内部验证 Macro F1、外部测试 Macro F1、optimism_gap 和 track_id 重叠比例。",
        "",
        "二、数据路径",
        f"读取数据：{DATA_PATH.as_posix()}",
        "",
        "三、字段处理",
        "目标变量：label",
        "分组字段：track_id",
        "删除的非模型字段：",
        ", ".join(dropped_columns),
        "",
        "四、final_test 划分方式",
        "先在 track_id 级别构建数据表，每个 track_id 只保留一条 label。",
        "再使用 train_test_split(test_size=0.2, random_state=42, stratify=track_table['label'])。",
        "final_test_track_ids 对应的所有 3 秒片段进入 final_test，其余进入 train_pool。",
        "final_test 不参与任何内部验证、模型选择或训练过程中的评估。",
        "",
        "五、样本数量",
        f"train_pool 样本数量：{len(train_pool)}",
        f"final_test 样本数量：{len(final_test)}",
        f"train_pool track_id 数量：{train_pool['track_id'].nunique()}",
        f"final_test track_id 数量：{len(final_test_track_ids)}",
        "",
        "六、类别分布",
        "train_pool 中每个类别的样本数量：",
        format_label_distribution(train_pool),
        "",
        "final_test 中每个类别的样本数量：",
        format_label_distribution(final_test),
        "",
        "七、每个模型、每种验证方式的结果摘要",
        results_df.to_string(index=False),
        "",
        "八、结果说明",
        "optimism_gap = internal_cv_macro_f1_mean - external_test_macro_f1。",
        "如果 optimism_gap 较大，说明内部验证结果相对真实未见歌曲测试表现更偏乐观。",
        "cv_track_id_overlap_ratio_mean 表示验证折样本中 track_id 同时出现在训练折中的比例。",
        "StratifiedKFold 按 3 秒片段随机划分，可能出现同一首歌的不同片段跨训练折和验证折。",
        "GroupKFold 按 track_id 分组划分，理论上 track_id 重叠比例应为 0，更接近未见歌曲测试场景。",
        "",
        "九、输出文件",
        f"文字日志：{SUMMARY_PATH.as_posix()}",
        f"最终测试集 track_id：{FINAL_TEST_TRACK_IDS_PATH.as_posix()}",
        f"CSV 结果表：{RESULTS_CSV_PATH.as_posix()}",
        f"内部/外部 F1 对比图：{INTERNAL_EXTERNAL_F1_IMAGE_PATH.as_posix()}",
        f"optimism_gap 对比图：{OPTIMISM_GAP_IMAGE_PATH.as_posix()}",
        f"track_id 重叠比例对比图：{TRACK_OVERLAP_IMAGE_PATH.as_posix()}",
        f"外部测试 Macro F1 对比图：{EXTERNAL_F1_IMAGE_PATH.as_posix()}",
        "",
        "实验完成。",
    ]

    return "\n".join(summary_lines)


def main():
    """执行划分方式可靠性实验，并保存日志、结果表和图像。"""
    data_file = PROJECT_ROOT / DATA_PATH
    log_dir = PROJECT_ROOT / LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file, encoding="utf-8")
    train_pool, final_test, final_test_track_ids = split_final_test(df)

    final_test_ids_text = "\n".join(final_test_track_ids) + "\n"
    (PROJECT_ROOT / FINAL_TEST_TRACK_IDS_PATH).write_text(
        final_test_ids_text,
        encoding="utf-8",
    )

    results_df, dropped_columns = run_experiment(train_pool, final_test)
    results_df.to_csv(
        PROJECT_ROOT / RESULTS_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    plot_files = save_plots(results_df)

    summary_log = build_summary_log(
        results_df=results_df,
        dropped_columns=dropped_columns,
        train_pool=train_pool,
        final_test=final_test,
        final_test_track_ids=final_test_track_ids,
    )
    (PROJECT_ROOT / SUMMARY_PATH).write_text(summary_log, encoding="utf-8")

    output_files = [
        SUMMARY_PATH,
        FINAL_TEST_TRACK_IDS_PATH,
        RESULTS_CSV_PATH,
        *plot_files,
    ]

    print("划分方式可靠性实验完成。输出文件如下：")
    for output_file in output_files:
        print(output_file.as_posix())


if __name__ == "__main__":
    main()
