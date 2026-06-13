# iListen Streamlit 在线部署说明

目标：把当前 Streamlit demo 部署成一个别人可以打开的网页链接。

## 推荐平台

优先使用 Streamlit Community Cloud。

原因：本项目是 Python + Streamlit + librosa + sklearn 模型文件，和 Streamlit Cloud 的运行方式最匹配。

## 部署前确认

需要提交到 GitHub 的关键文件：

```text
streamlit_demo/streamlit_app.py
streamlit_demo/audio_feature_extractor.py
streamlit_demo/model_predictor.py
streamlit_demo/requirements.txt
streamlit_demo/model_artifacts/
streamlit_demo/assets/
data/processed/processed_features_3_sec.csv
packages.txt
```

当前不建议提交：

```text
streamlit_demo/local_samples/
```

原因：这里如果放商业歌曲片段，公开部署会有版权风险。线上版本可以先保留上传功能；如果要放样例，建议换成可公开使用或小组有授权的音频片段。

## Streamlit Cloud 填写信息

在 Streamlit Cloud 新建 App 时填写：

```text
Repository: 3271285053-star/music-genre-classification
Branch: main
Main file path: streamlit_demo/streamlit_app.py
```

Python 版本建议选择：

```text
Python 3.11
```

依赖文件位置：

```text
streamlit_demo/requirements.txt
```

系统依赖：

```text
packages.txt
```

## 部署后检查

打开生成的网页链接后检查：

```text
1. 首页能正常显示 iListen 风格页面
2. 上传 MP3 / WAV 能正常分析
3. 上传 MP4 能正常提取音频
4. 能输出 GTZAN 十类风格雷达
5. 能输出我锐评分数、三条听感轴和锐评卡
6. 导出锐评卡 PNG 按钮能用
```

## 如果部署失败

优先看 Streamlit Cloud 的日志。

常见问题：

```text
ModuleNotFoundError
说明 requirements.txt 缺依赖。

FileNotFoundError
说明模型文件、assets 或 data/processed/processed_features_3_sec.csv 没提交。

ffmpeg 相关错误
检查 packages.txt 是否在仓库根目录，并包含 ffmpeg。

样例音频不显示
这是正常的，因为 local_samples 默认不提交。线上先用上传文件测试。
```
