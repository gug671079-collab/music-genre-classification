# Streamlit 音乐流派分类 Demo

本目录是课程展示用的网页 demo。主线使用 Streamlit，不依赖 React/Vite。

## 功能

- 上传音频文件：`wav`、`mp3`、`flac`、`ogg`、`m4a`
- 预留视频输入：`mp4`、`mov`、`m4v`，通过 ffmpeg 提取音轨
- 使用 librosa 提取 GTZAN 风格 3 秒切片特征
- 调用 sklearn SVM-RBF 模型输出十类音乐流派预测
- 展示波形、核心参数、特征时间变化、Top-3 概率、参数表和特征解释

GTZAN 十类：

```text
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
```

## 安装依赖

在项目根目录 `repo_tmp_music_genre` 下运行：

```powershell
pip install -r streamlit_demo\requirements.txt
```

## 生成模型工件

默认使用当前仓库数据：

```text
data/processed/processed_features_3_sec.csv
```

运行：

```powershell
python streamlit_demo\build_demo_model_artifacts.py
```

输出位置：

```text
streamlit_demo/model_artifacts
```

说明：这里生成的是课程演示用 sklearn 工件。如果组员后续给出最终训练模型，可以替换该目录下的 `classifier.joblib`、`feature_columns.json`、`class_names.json`。

## 启动网页

```powershell
streamlit run streamlit_demo\streamlit_app.py
```

浏览器打开 Streamlit 给出的本地地址即可。

## 文件说明

```text
streamlit_app.py                 Streamlit 网页入口
audio_feature_extractor.py       音频/视频音轨读取与特征提取
model_predictor.py               sklearn 模型加载、分段预测、概率聚合
build_demo_model_artifacts.py    从当前 CSV 训练演示模型工件
requirements.txt                 网页 demo 依赖
```
