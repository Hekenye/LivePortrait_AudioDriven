# LivePortrait: Audio-Driven Talking Head Version 🎙️

## 🔍 项目概述

本项目基于 [LivePortrait](https://github.com/KwaiVGI/LivePortrait) —— 一个高效的图像驱动式人像动画生成方法，支持通过输入驱动图像来控制源图像的表情和姿态。我们在此基础上进行了扩展，开发了一个 **音频驱动版本**，实现了从语音到动态人像的实时生成。


### ✨ 核心工作

我们的**音频驱动版本的 LivePortrait**，可以实现从语音信号直接生成人脸动画，具体改进包括：

- **音频编码模块**：引入了音频编码器（Whisper），将输入语音转换为时序特征表示。
- **关键点预测**：以流匹配（FlowMatching）的方式，训练了一个轻量级的Transformer，根据时序音频特征条件，生成对应面部动作的关键点运动序列（关键点的定义请参照论文第三节：[arxiv](https://arxiv.org/pdf/2407.03168)）。
- **推理结构**：在保持低延迟的前提下，将关键点预测网络与LivePortrait预训练组件相结合，实现实时音频驱动动画生成。

### 🎯 应用场景

该音频驱动版本可用于以下应用：

- 数字人语音交互
- 动画配音与自动口型匹配
- AI 客服生成等

---

## 🧰 使用说明

本项目依赖于原始 LivePortrait 的模型结构与部分代码库，请先确保已正确安装并配置好环境。（安装环境部分说明来自LivePortrait原仓库）

### 1. 克隆代码和安装运行环境 🛠️

> [!Note]
> 确保您的系统已安装[`git`](https://git-scm.com/)、[`conda`](https://anaconda.org/anaconda/conda)和[`FFmpeg`](https://ffmpeg.org/download.html)。有关FFmpeg安装的详细信息，见[**如何安装FFmpeg**](assets/docs/how-to-install-ffmpeg.md)。

```bash
git clone https://github.com/KwaiVGI/LivePortrait
cd LivePortrait

# 使用conda创建环境
conda create -n LivePortrait python=3.10
conda activate LivePortrait
```

首先，通过以下命令检查您当前的CUDA版本：

```bash
nvcc -V # example versions: 11.1, 11.8, 12.1, etc.
```

然后，安装相应版本的torch。以下是不同CUDA版本的示例。如果您的CUDA版本未列出，请访问[PyTorch官方网站](https://pytorch.org/get-started/previous-versions)获取安装命令：
```bash
# for CUDA 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# for CUDA 11.8
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
# for CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
# ...
```


最后，安装其余依赖项：

```bash
pip install -r requirements.txt
```


### 2. 下载预训练的LivePortriat组件权重 📥

从HuggingFace下载预训练权重的最简单方法是：
```bash
# !pip install -U "huggingface_hub[cli]"
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```

若您不能访问HuggingFace平台，你可以访问其镜像网站[hf-mirror](https://hf-mirror.com/)进行下载操作：

```bash
# !pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```

或者，您可以从[Google Drive](https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib)或[百度云](https://pan.baidu.com/s/1MGctWmNla_vZxDbEp2Dtzw?pwd=z5cn)（进行中）下载所有预训练权重。解压并将它们放置在`./pretrained_weights`目录下。


### 3. 下载音频编码器权重（Whisper）
首先下载Whispe_tiny权重文件：[Whisper](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)

然后在`./pretrained_weights`目录下创建`audio_processor`子目录，将权重文件放在`./pretrained_weights/audio_processor`目录下

确保目录结构如所示包含[**本仓库该路径**](assets/docs/directory-structure.md)其中展示的内容。


### 4. 训练模型
本项目所使用的训练数据集为MEAD，是一个包含 60 名演员在八种不同情绪（不含中性）的三种不同强度下进行对话的面部视频数据集。如果您有需要，可以参照[MEAD](https://wywu.github.io/projects/MEAD/MEAD.html)获取更详细的信息。

在下载数据集后，可以通过process_data_MEAD.py来对MEAD进行处理，得到训练数据集：
```bash
python process_data_MEAD.py  \
  --dataset_path your_path/MEAD  \        # MEAD原始数据路径
  --output_path your_path/processed_MEAD  # 处理数据的保存路径
```

之后便可以启动训练代码：
```bash
python train.py -dataset_path your_path/processed_MEAD --work_dir output
```
您可以参照训练参数文件，实现更多参数的调节[config](src/config/train_config.py)

### 5. 推理

使用推理代码，加载预训练的模型，可以轻松获得音频驱动下的人脸动画。

您需要指定`--statistic_path`来确定模型预测时所使用的数据分布均值和方差，如果您使用process_data_MEAD.py处理数据集，则对应的统计文件路径为`.../processed_MEAD/statistic.pt`。

您还需要指定`--pretrained_model_path`来加载预训练的模型。

其它参数的含义：

- `-s`：参考人脸图像路径
- `-d`：驱动音频路径
- `-o`：生成结果存放路径
- `seed`：随机种子（可选，默认为0）
- `cfg_scale`：引导系数（可选，默认为4.5）
- `output_fps`：输出视频帧率（可选，默认为30）

```bash
python inference_with_audio.py \
  -s assets/examples/source/s9.jpg \
  -d assets/examples/driving/angry.m4a \
  -o generated \
  --statistic_path your_statistic_path \
  --pretrained_model_path your_model_path
```


## 致谢
我们非常感谢[LivePortrait](https://github.com/KwaiVGI/LivePortrait), [Whisper](https://github.com/openai/whisper)和[MMAudio](https://github.com/hkchengrex/MMAudio)，感谢他们的开放研究和贡献。
