import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 读取数据
df = pd.read_csv(r"D:\Music AI\music-genre-classification\data\processed\processed_features_3_sec.csv")

target_column = 'label' # 标签列名
X = df.drop(columns=[target_column]) # 特征数据
y = df[target_column]                # 预测目标

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)# 标准化后的训练数据
X_test_scaled = scaler.transform(X_test)# 标准化后的测试数据

# 训练AI
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# 评估准确率
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)# 计算准确率

print(f"模型准确率: {accuracy:.4f}")
print("详细分类报告:")
print(classification_report(y_test, y_pred))