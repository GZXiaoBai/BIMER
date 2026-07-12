# Kaggle 特征提取指南

## 1. 数据授权

1. 在 Hugging Face 登录账号。
2. 打开 `BAAI/Emotiontalk`，阅读并接受共享联系信息及非商业学术条款。
3. 创建只读 Token，保存到 Kaggle Secret `HF_TOKEN`，不要写进 Notebook 或仓库。
4. MELD 使用 `declare-lab/MELD` 官方数据入口。

EmotionTalk 约36.1 GB。建议把原始数据、标准特征及每组鲁棒性特征分别保存为私有 Kaggle Dataset，避免会话结束后丢失。

## 2. 安装

```bash
git clone <your-repository-url> /kaggle/working/bimer
cd /kaggle/working/bimer
pip install -e '.[inference]'
./scripts/download_yunet.sh /kaggle/working/yunet.onnx
```

Kaggle通常已安装匹配CUDA的PyTorch；若版本冲突，优先保留Kaggle自带的 `torch/torchvision`，再单独安装其他依赖。

## 3. 下载

```bash
hf download declare-lab/MELD --repo-type dataset --local-dir /kaggle/working/meld
hf download BAAI/Emotiontalk --repo-type dataset --local-dir /kaggle/working/emotiontalk
```

如访问 EmotionTalk 返回403，说明账号尚未完成数据授权，不要绕过授权使用第三方镜像。

## 4. 分阶段提取

先生成并校验统一清单，再分别按数据集和划分提取，以控制单次会话时长：

```bash
bimer extract-features \
  --manifest /kaggle/working/all.jsonl \
  --features /kaggle/working/features-standard \
  --yunet-model /kaggle/working/yunet.onnx \
  --dataset meld --split train --device cuda
```

依次处理 `meld/{train,dev,test}` 与 `emotiontalk/{train,validation,test}`。每处理完一个划分，将目录发布为私有 Kaggle Dataset。

## 5. 资源策略

- 文本、音频、视频编码器顺序加载，完成一个阶段后释放模型并重启 Session。
- 标准特征只提取一次；融合模型训练直接读取 `.npz`，无需GPU也可运行。
- 加噪和丢帧必须发生在预训练编码器之前，单独写入新的特征根目录。
- 每个特征根目录运行 `bimer validate` 和合成测试后再发布。

