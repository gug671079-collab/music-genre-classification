import pandas as pd

# ===== 读取数据集原始csv文件 =====
df = pd.read_csv("data/raw/features_3_sec.csv")

print(df.info())

# ===== 检查缺失值 =====
print(df.isnull().sum())

# ===== 检查重复值 =====
duplicates = df.duplicated().sum()

print("重复行数量:", duplicates)

if duplicates > 0 :
    df = df.drop_duplicates()
    print(df.shape)

# ===== 检查类别分布 =====
print(df["label"].value_counts())

# ===== 检查异常数据（初步） =====
print(df.describe())

# ===== 删去无用列（filename） =====
df = df.drop(columns=["filename"])

# ===== 保存processed数据 =====
df.to_csv("data/processed/processed_features_3_sec.csv", index=False)

print("数据清洗完成！")