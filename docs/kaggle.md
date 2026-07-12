# Kaggle 特征提取指南

## 1. 数据授权

1. 在 Hugging Face 登录账号。
2. 打开 `BAAI/Emotiontalk`，阅读并接受共享联系信息及非商业学术条款。
3. 创建只读 Token，保存到 Kaggle Secret `HF_TOKEN`，不要写进 Notebook 或仓库。
4. MELD 使用 `declare-lab/MELD` 官方数据入口。

EmotionTalk 整库约36.1 GB；本项目只需要其中约21.3 GB的 `Multimodal.tar`。MP4 中的音轨直接供语音编码器使用，因此不重复下载14.8 GB的 `Audio.tar`。Kaggle 的 `/kaggle/working` 持久化上限低于该归档的实际大小，脚本会把归档和解压媒体放在 `/tmp/bimer-data`，只把清单和特征写入 `/kaggle/working`。原始媒体不会跨 Session 保留，因此下载后应在同一 Session 中完成特征提取。

## 2. 安装

```bash
git clone <your-repository-url> /kaggle/working/bimer
cd /kaggle/working/bimer
pip install -e '.[inference]'
./scripts/download_yunet.sh /kaggle/working/yunet.onnx
```

Kaggle通常已安装匹配CUDA的PyTorch；若版本冲突，优先保留Kaggle自带的 `torch/torchvision`，再单独安装其他依赖。

## 3. 读取 Kaggle Secret

在 Notebook 的 Python 单元格执行：

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

该代码不会打印 Token。不要使用 `print` 检查密钥。

## 4. 一键下载、解压、生成清单和校验

```bash
./scripts/prepare_emotiontalk_kaggle.sh
```

脚本会：

1. 只下载 `BAAI/Emotiontalk` 的 `Multimodal.tar`。
2. 解压逐句 MP4。
3. 锁定官方 `NKU-HLT/EmotionTalk` 仓库提交，读取 `mm.csv` 与 `transcription.csv`。
4. 按官方组别产生 `train=15413`、`validation=1908`、`test=1929` 的 JSONL 清单。
5. 执行样本量、标签和跨划分重复校验。

默认原始数据目录为 `/tmp/bimer-data`，清单输出为 `/kaggle/working/bimer-output/emotiontalk.jsonl`。

## 5. 手动下载（排错用）

```bash
hf download declare-lab/MELD --repo-type dataset --local-dir /kaggle/working/meld
hf download BAAI/Emotiontalk Multimodal.tar \
  --repo-type dataset \
  --local-dir /kaggle/working/emotiontalk
```

如访问 EmotionTalk 返回403，说明账号尚未完成数据授权，不要绕过授权使用第三方镜像。

## 6. 双 T4 并行提取（推荐）

Kaggle 加速器选择 `GPU T4 x2`。先运行 validation 小规模检查，再运行完整划分。优化流水线把文本和语音放在 GPU 0、视觉放在 GPU 1，音视频解码与 YuNet 检测使用 CPU 工作进程：

```bash
bimer extract-features \
  --manifest /kaggle/working/bimer-output/emotiontalk.jsonl \
  --features /kaggle/working/features-emotiontalk-validation-v2 \
  --staging /kaggle/working/features-emotiontalk-validation-v2 \
  --yunet-model /kaggle/working/yunet.onnx \
  --dataset emotiontalk --split validation \
  --mode parallel \
  --text-audio-device cuda:0 \
  --vision-device cuda:1 \
  --text-batch-size 64 \
  --audio-batch-size 8 \
  --vision-batch-size 8 \
  --audio-workers 4 \
  --vision-workers 4 \
  --queue-capacity 8 \
  --shard-size 16
```

在另一个终端查看两张 GPU 的实时负载：

```bash
watch -n 2 nvidia-smi
```

查看各模态 staging 和最终分片数量：

```bash
find /kaggle/working/features-emotiontalk-validation-v2/staging \
  -name 'features-*.npz' | sort | tail
find /kaggle/working/features-emotiontalk-validation-v2/emotiontalk/validation \
  -name 'features-*.npz' | wc -l
```

每个 staging 分片先写临时文件，通过 ID、维度和有限值校验后才原子发布。Session 中断后运行完全相同的命令即可续跑；已校验的文本、语音、视觉或最终分片不会重复计算。优化版通过32样本检查前，保留旧任务和旧特征目录，不覆盖或删除。

如果并行流程出现无法恢复的问题，使用原串行路径回退：

```bash
bimer extract-features \
  --manifest /kaggle/working/bimer-output/emotiontalk.jsonl \
  --features /kaggle/working/features-emotiontalk-validation-serial \
  --yunet-model /kaggle/working/yunet.onnx \
  --dataset emotiontalk --split validation \
  --mode serial --device cuda --shard-size 16
```

验证完成后，依次处理 `meld/{train,dev,test}` 与 `emotiontalk/{train,validation,test}`。不同数据集和划分使用不同特征根目录，并在完成后发布为私有 Kaggle Dataset。

## 7. EmotionTalk train 跨 Session 分段提取

train 共15,413条，使用 `--shard-size 16` 时为964个全局分片。按左闭右开区间依次运行：

```text
[0, 120) [120, 240) [240, 360) [360, 480)
[480, 600) [600, 720) [720, 840) [840, 964)
```

每个区间沿用同一个目录 `/kaggle/working/features-emotiontalk-train-v4`。例如第一段：

```bash
bimer extract-features \
  --manifest /kaggle/working/bimer-output/emotiontalk-feature.jsonl \
  --features /kaggle/working/features-emotiontalk-train-v4 \
  --staging /kaggle/working/features-emotiontalk-train-v4 \
  --yunet-model /kaggle/working/yunet.onnx \
  --dataset emotiontalk --split train --mode parallel \
  --text-audio-device cuda:0 --vision-device cuda:1 \
  --text-batch-size 64 --audio-batch-size 8 --vision-batch-size 8 \
  --audio-workers 4 --vision-workers 4 --queue-capacity 8 \
  --shard-size 16 --start-shard 0 --end-shard 120
```

完成后必须验证并写入完成标记：

```bash
bimer verify-features \
  --manifest /kaggle/working/bimer-output/emotiontalk-feature.jsonl \
  --features /kaggle/working/features-emotiontalk-train-v4 \
  --dataset emotiontalk --split train --shard-size 16 \
  --start-shard 0 --end-shard 120 --write-completion
```

成功后会生成 `ranges/range-00000-00120.json`。此文件存在且内容中 `is_valid=true` 后，使用 Kaggle **Quick Save** 保存版本，再进入下一段。

新 Session 启动时，先从最近一次已保存的 Notebook Output 恢复同一目录：

```python
from pathlib import Path
import shutil

saved = Path(
    "/kaggle/input/notebooks/zhoujunjie2/"
    "bimer-emotiontalk-bootstrap/features-emotiontalk-train-v4"
)
working = Path("/kaggle/working/features-emotiontalk-train-v4")
if saved.is_dir():
    shutil.copytree(saved, working, dirs_exist_ok=True)
```

恢复后先验证上一段，再开始下一段。最后一段使用：

```bash
--start-shard 840 --end-shard 964
```

八段全部完成后执行不带范围的全量验收：

```bash
bimer verify-features \
  --manifest /kaggle/working/bimer-output/emotiontalk-feature.jsonl \
  --features /kaggle/working/features-emotiontalk-train-v4 \
  --dataset emotiontalk --split train --shard-size 16
```

最终必须报告 `samples=15413`、`verified_shards=964`、`start_shard=0`、`end_shard=964`。范围参数只允许用于 `--mode parallel`，且必须同时指定dataset、split、start和end。重复执行同一区间会验证并跳过已有最终分片；中断时可用完全相同的命令续跑。

## 8. 资源策略

- 三个预训练编码器全部冻结；GPU 0 先运行文本再运行语音，GPU 1 同时运行视觉。
- GPU 批量由 `--text-batch-size`、`--audio-batch-size` 和 `--vision-batch-size` 控制；遇到 CUDA OOM 会逐次减半，最低降到1。
- `--shard-size` 只控制 staging 和最终落盘分片，不代表 GPU 批量。
- 音频和视觉各使用4个 CPU 工作进程及有界预取队列，避免一次性把全部波形或视频 clip 放入内存。
- 特征提取会校验并跳过已存在的同编号分片；Kaggle Session 中断后可用相同命令安全续跑。
- 标准特征只提取一次；融合模型训练直接读取 `.npz`，无需GPU也可运行。
- 加噪和丢帧必须发生在预训练编码器之前，单独写入新的特征根目录。
- 每个特征根目录运行 `bimer validate` 和合成测试后再发布。
