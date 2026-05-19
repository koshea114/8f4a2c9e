# 手机端拍照鸡只数量检测（YOLO）

本仓库新增了一个最小可运行实现：使用 YOLO 对拍照图片进行目标检测，并输出画面中的鸡数量。

## 文件说明

- `chicken_detector.py`：核心检测脚本
- `tests/test_chicken_detector.py`：鸡数量统计逻辑的单元测试

## 本地运行

```bash
pip install ultralytics opencv-python
python chicken_detector.py --image /absolute/path/to/photo.jpg --output /absolute/path/to/result.jpg
```

运行后会输出：

- `chicken_count=<数量>`（始终输出）
- `saved_result=<标注后图片路径>`（仅在传入 `--output` 时输出）

## 手机部署建议

可将该逻辑部署到手机端时采用以下方式之一：

1. 将 YOLO 模型导出为 ONNX / TFLite 并接入 Android/iOS 客户端；
2. 在 App 中调用相机“拍照”后，把图片传给检测模块执行 `detect_chickens_in_image`；
3. 使用自定义鸡数据集训练模型时，脚本会优先统计 `chicken/hen/rooster` 类；若使用 COCO 通用模型，则可回退统计 `bird` 类。
