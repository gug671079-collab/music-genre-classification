from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ===== 路径设置 =====
# 本文件位于 src 目录下，parents[1] 可以定位到项目根目录。
# 下面只使用项目内的相对路径，不写死任何本地绝对路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path("data") / "processed" / "processed_features_3_sec_V2.csv"
LOG_PATH = Path("logs") / "baseline_v2_summary.txt"
CONFUSION_MATRIX_IMAGE_PATH = Path("logs") / "baseline_v2_confusion_matrix.png"


def save_confusion_matrix_image(matrix, labels, output_path):
    """把混淆矩阵保存为图片；如果环境缺少绘图库，则跳过图片输出。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False, "未安装 matplotlib，跳过混淆矩阵图片输出。"

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_title("Baseline V2 Confusion Matrix")
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

    return True, f"混淆矩阵图片已保存到：{output_path.relative_to(PROJECT_ROOT).as_posix()}"


def build_summary_log(
    data_path,
    dropped_columns,
    feature_count,
    train_size,
    test_size,
    knn_params,
    accuracy,
    macro_precision,
    macro_recall,
    macro_f1,
    report,
    matrix,
    labels,
    image_message,
):
    """整理模型训练和评估结果，生成日志文本。"""
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    summary_lines = [
        "音乐风格自动分类系统 - KNN Baseline V2",
        "=" * 50,
        "",
        "一、数据文件",
        f"读取数据：{data_path.as_posix()}",
        "",
        "二、数据字段处理",
        "目标变量：label",
        "删除的非模型字段：",
        ", ".join(dropped_columns),
        f"最终用于训练的特征数量：{feature_count}",
        "",
        "三、train/test 划分",
        f"训练集比例：{train_size:.1f}",
        f"测试集比例：{test_size:.1f}",
        "划分方式：train_test_split(test_size=0.3, random_state=42, stratify=y)",
        "",
        "四、KNN 参数",
        f"k 值 n_neighbors：{knn_params['n_neighbors']}",
        f"weights：{knn_params['weights']}",
        f"metric：{knn_params['metric']}",
        f"p：{knn_params['p']}",
        "",
        "五、评价指标",
        f"Accuracy：{accuracy:.4f}",
        f"Macro Precision：{macro_precision:.4f}",
        f"Macro Recall：{macro_recall:.4f}",
        f"Macro F1：{macro_f1:.4f}",
        "",
        "六、classification_report",
        report,
        "",
        "七、confusion matrix 数值结果",
        matrix_df.to_string(),
        "",
        "八、可选图像输出",
        image_message,
        "",
        "训练与评估完成。",
    ]

    return "\n".join(summary_lines)


def main():
    """执行数据读取、字段处理、模型训练、模型评估和日志输出。"""
    data_file = PROJECT_ROOT / DATA_PATH
    log_file = PROJECT_ROOT / LOG_PATH
    confusion_matrix_image_file = PROJECT_ROOT / CONFUSION_MATRIX_IMAGE_PATH

    # 自动创建 logs 文件夹，避免保存日志和图片时报错。
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # 读取 V2 清洗后的数据，不修改任何原始数据或清洗后数据。
    df = pd.read_csv(data_file, encoding="utf-8")

    if "label" not in df.columns:
        raise ValueError("数据中缺少目标字段 label，无法训练模型。")

    # label 是预测目标 y；其他可用的数值音频特征作为模型输入 X。
    y = df["label"]
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
    X = df.drop(columns=existing_non_model_columns + ["label"])
    X = X.select_dtypes(include="number")

    if X.empty:
        raise ValueError("删除非模型字段后没有可用于训练的数值音频特征。")

    # 使用 stratify=y 保证训练集和测试集中的类别比例尽量一致。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # 用 Pipeline 把标准化和 KNN 模型串起来，避免手动处理训练集/测试集标准化。
    knn_params = {
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "minkowski",
        "p": 2,
    }
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(**knn_params)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels = sorted(y.unique())
    accuracy = accuracy_score(y_test, y_pred)
    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    image_saved, image_message = save_confusion_matrix_image(
        matrix,
        labels,
        confusion_matrix_image_file,
    )

    summary_log = build_summary_log(
        data_path=DATA_PATH,
        dropped_columns=existing_non_model_columns,
        feature_count=X.shape[1],
        train_size=0.7,
        test_size=0.3,
        knn_params=knn_params,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        report=report,
        matrix=matrix,
        labels=labels,
        image_message=image_message,
    )
    log_file.write_text(summary_log, encoding="utf-8")

    print("Baseline V2 训练与评估完成。")
    print(f"训练日志已保存到：{LOG_PATH.as_posix()}")
    if image_saved:
        print(f"混淆矩阵图片已保存到：{CONFUSION_MATRIX_IMAGE_PATH.as_posix()}")
    else:
        print(image_message)


if __name__ == "__main__":
    main()
