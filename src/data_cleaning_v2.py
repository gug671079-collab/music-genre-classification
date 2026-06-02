from pathlib import Path

import pandas as pd


# ===== 路径设置 =====
# 本文件位于 src 目录下，parents[1] 可以定位到项目根目录。
# 这里没有写死电脑上的绝对路径，只把项目内的相对路径拼接起来使用。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = Path("data") / "raw" / "features_3_sec.csv"
PROCESSED_DATA_PATH = Path("data") / "processed" / "processed_features_3_sec_V2.csv"
LOG_PATH = Path("logs") / "data_cleaning_summary.txt"


def parse_filename(filename):
    """从 filename 中提取音乐风格、音轨编号和 3 秒片段编号。"""
    stem_parts = Path(str(filename)).stem.split(".")

    if len(stem_parts) < 3:
        return pd.Series(
            {
                "genre_from_filename": pd.NA,
                "track_id": pd.NA,
                "segment_id": pd.NA,
            }
        )

    genre = stem_parts[0]
    track_number = stem_parts[1]
    segment_id = stem_parts[2]

    return pd.Series(
        {
            "genre_from_filename": genre,
            "track_id": f"{genre}.{track_number}",
            "segment_id": segment_id,
        }
    )


def series_to_text(series):
    """把 Series 转成适合写入日志的文本，避免空结果时显示不清楚。"""
    if series.empty:
        return "无"
    return series.to_string()


def build_summary_log(
    df_raw,
    df_with_metadata,
    df_cleaned,
    required_columns,
    missing_required_columns,
    full_duplicate_count,
    model_duplicate_count,
    multi_label_feature_count,
    multi_label_examples,
    model_columns,
):
    """整理清洗过程中的关键信息，生成日志文本。"""
    label_match = df_with_metadata["label"] == df_with_metadata["genre_from_filename"]
    mismatch_rows = df_with_metadata.loc[
        ~label_match,
        ["filename", "label", "genre_from_filename", "track_id", "segment_id"],
    ]

    track_segment_counts = (
        df_with_metadata.groupby("track_id")["segment_id"]
        .count()
        .sort_index()
    )
    track_segment_distribution = track_segment_counts.value_counts().sort_index()

    summary_lines = [
        "音乐风格自动分类系统 - 数据清洗日志 V2",
        "=" * 50,
        "",
        "一、输入输出路径",
        f"读取原始数据：{RAW_DATA_PATH.as_posix()}",
        f"输出清洗数据：{PROCESSED_DATA_PATH.as_posix()}",
        f"输出清洗日志：{LOG_PATH.as_posix()}",
        "",
        "二、数据基本信息",
        f"原始数据行数：{df_raw.shape[0]}",
        f"原始数据列数：{df_raw.shape[1]}",
        f"必要字段检查：{', '.join(required_columns)}",
        f"缺失的必要字段：{', '.join(missing_required_columns) if missing_required_columns else '无'}",
        "字段名：",
        ", ".join(df_raw.columns),
        "",
        "字段数据类型：",
        df_raw.dtypes.to_string(),
        "",
        "三、缺失值检查",
        series_to_text(df_raw.isnull().sum()),
        "",
        "四、label 类别分布",
        series_to_text(df_raw["label"].value_counts().sort_index()),
        "",
        "五、filename 信息提取",
        "新增字段：genre_from_filename、track_id、segment_id",
        f"genre_from_filename 缺失数量：{df_with_metadata['genre_from_filename'].isnull().sum()}",
        f"track_id 缺失数量：{df_with_metadata['track_id'].isnull().sum()}",
        f"segment_id 缺失数量：{df_with_metadata['segment_id'].isnull().sum()}",
        "",
        "六、label 与 filename 中风格是否一致",
        f"一致记录数：{label_match.sum()}",
        f"不一致记录数：{len(mismatch_rows)}",
        "不一致样例：",
        mismatch_rows.head(20).to_string(index=False) if not mismatch_rows.empty else "无",
        "",
        "七、track_id 与切片数量摘要",
        f"track_id 总数：{track_segment_counts.shape[0]}",
        "每个 track_id 的片段数量分布（片段数：track_id 数量）：",
        series_to_text(track_segment_distribution),
        "",
        "八、重复值检查",
        f"包含 filename 时的完整重复行数量：{full_duplicate_count}",
        "用于检查“特征 + label”重复的字段：",
        ", ".join(model_columns),
        f"排除 filename、genre_from_filename、track_id、segment_id、length 后的重复记录数量：{model_duplicate_count}",
        "",
        "相同音频特征是否对应多个 label：",
        f"存在多个 label 的相同音频特征组数量：{multi_label_feature_count}",
        "问题样例：",
        multi_label_examples.to_string(index=False) if not multi_label_examples.empty else "无",
        "",
        "九、去重结果",
        f"去重前行数：{df_with_metadata.shape[0]}",
        f"去重后行数：{df_cleaned.shape[0]}",
        f"删除重复行数：{df_with_metadata.shape[0] - df_cleaned.shape[0]}",
        "",
        "十、最终输出数据",
        f"最终输出行数：{df_cleaned.shape[0]}",
        f"最终输出列数：{df_cleaned.shape[1]}",
        "保留字段说明：保留 filename、track_id、segment_id，方便后续按音轨分组划分训练集和测试集。",
        "",
        "清洗完成。",
    ]

    return "\n".join(summary_lines)


def main():
    """执行数据读取、信息提取、重复值检查、去重和日志输出。"""
    raw_data_file = PROJECT_ROOT / RAW_DATA_PATH
    processed_data_file = PROJECT_ROOT / PROCESSED_DATA_PATH
    log_file = PROJECT_ROOT / LOG_PATH

    # 自动创建输出目录，避免保存文件时报错。
    processed_data_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # 只读取原始 features_3_sec.csv，不修改原始数据文件。
    df_raw = pd.read_csv(raw_data_file, encoding="utf-8")

    # 检查后续清洗必须依赖的字段是否存在，避免字段缺失时产生错误结果。
    required_columns = ["filename", "label"]
    missing_required_columns = [
        column for column in required_columns if column not in df_raw.columns
    ]
    if missing_required_columns:
        raise ValueError(
            "原始数据缺少必要字段："
            + ", ".join(missing_required_columns)
        )

    # 第一步：检查包含 filename 时的完整重复行数量。
    full_duplicate_count = df_raw.duplicated().sum()

    # 保留 filename，并从 filename 中提取后续分组划分需要的信息。
    extracted_metadata = df_raw["filename"].apply(parse_filename)
    df_with_metadata = pd.concat([df_raw, extracted_metadata], axis=1)

    # segment_id 表示 0 到 9 的片段编号，转成整数更方便后续统计。
    df_with_metadata["segment_id"] = pd.to_numeric(
        df_with_metadata["segment_id"],
        errors="coerce",
    ).astype("Int64")

    # 第二步：排除元信息字段，只检查“特征 + label”是否重复。
    metadata_columns = [
        "filename",
        "length",
        "genre_from_filename",
        "track_id",
        "segment_id",
    ]
    model_columns = [
        column for column in df_with_metadata.columns if column not in metadata_columns
    ]
    model_duplicate_count = df_with_metadata.duplicated(subset=model_columns).sum()

    # 检查“相同音频特征是否对应多个 label”。
    # 这里只记录日志，不自动删除，因为多标签问题需要人工判断原因。
    audio_feature_columns = [
        column for column in model_columns if column != "label"
    ]
    label_counts_by_features = (
        df_with_metadata.groupby(audio_feature_columns, dropna=False)["label"]
        .nunique()
        .reset_index(name="label_count")
    )
    multi_label_features = label_counts_by_features[
        label_counts_by_features["label_count"] > 1
    ]
    multi_label_feature_count = len(multi_label_features)

    if multi_label_features.empty:
        multi_label_examples = pd.DataFrame()
    else:
        multi_label_examples = df_with_metadata.merge(
            multi_label_features[audio_feature_columns],
            on=audio_feature_columns,
            how="inner",
        )[["filename", "label", "genre_from_filename", "track_id", "segment_id"]].head(20)

    # 对模型训练数据中的完全重复记录进行去重，保留第一次出现的记录。
    df_cleaned = df_with_metadata.drop_duplicates(
        subset=model_columns,
        keep="first",
    ).reset_index(drop=True)

    # 保存清洗后的数据。输出文件保留 filename、track_id、segment_id。
    df_cleaned.to_csv(processed_data_file, index=False, encoding="utf-8")

    # 生成并保存清洗日志。
    summary_log = build_summary_log(
        df_raw=df_raw,
        df_with_metadata=df_with_metadata,
        df_cleaned=df_cleaned,
        required_columns=required_columns,
        missing_required_columns=missing_required_columns,
        full_duplicate_count=full_duplicate_count,
        model_duplicate_count=model_duplicate_count,
        multi_label_feature_count=multi_label_feature_count,
        multi_label_examples=multi_label_examples,
        model_columns=model_columns,
    )
    log_file.write_text(summary_log, encoding="utf-8")

    print("数据清洗完成。")
    print(f"清洗后数据已保存到：{PROCESSED_DATA_PATH.as_posix()}")
    print(f"清洗日志已保存到：{LOG_PATH.as_posix()}")


if __name__ == "__main__":
    main()
