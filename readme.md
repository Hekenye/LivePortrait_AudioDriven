# LivePortrait: Audio-Driven Talking Head Version 🎙️

<div align="center">
<strong>English</strong> | <a href="./readme_zh_cn.md"><strong>简体中文</strong></a>
</div>


## 🔍 Project Overview

This project is based on [LivePortrait](https://github.com/KwaiVGI/LivePortrait), an efficient image-driven portrait animation method that supports controlling the expression and pose of a source image using input driving images. We have extended this framework to develop an audio-driven version , enabling real-time generation of facial animation directly from speech.


### ✨ Core Contributions

Our **audio-driven version of LivePortrait** achieves speech-to-animation generation by making the following improvements:

- **Audio Encoding Module**: Integrate an audio encoder (Whisper) to convert input speech into temporal feature representations.
- **Key Point Prediction**: Train a lightweight Transformer using flow matching to generate sequences of facial motion key points conditioned on the temporal audio features（see Section 3 of the paper for key point definitions: [arxiv](https://arxiv.org/pdf/2407.03168)）.
- **Inference Architecture**: Combine the keypoint prediction network with pre-trained components of LivePortrait to achieve low-latency, real-time audio-driven animation generation.

### 🎯 Applications

The audio-driven version can be applied to various use cases, including:

- Digital human voice interaction
- Animated dubbing and automatic lip syncing
- AI customer service avatars

---

## 🧰 Usage Guide

This project relies on the model structure and part of the codebase of the original LivePortrait. Please ensure you have installed and configured the environment correctly first. (Installation instructions are adapted from the LivePortrait repository.)

### 1. Clone the Code and Set Up Environment 🛠️

> [!Note]
> Make sure your system has [`git`](https://git-scm.com/), [`conda`](https://anaconda.org/anaconda/conda), and [`FFmpeg`](https://ffmpeg.org/download.html) installed. For details on FFmpeg installation, see [**how to install FFmpeg**](assets/docs/how-to-install-ffmpeg.md).

```bash
git clone https://github.com/KwaiVGI/LivePortrait
cd LivePortrait

# create env using conda
conda create -n LivePortrait python=3.10
conda activate LivePortrait
```

Firstly, check your current CUDA version by:

```bash
nvcc -V # example versions: 11.1, 11.8, 12.1, etc.
```

Then, install the corresponding torch version. Here are examples for different CUDA versions. Visit the [PyTorch Official Website](https://pytorch.org/get-started/previous-versions) for installation commands if your CUDA version is not listed:
```bash
# for CUDA 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# for CUDA 11.8
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
# for CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
# ...
```


Finally, install the remaining dependencies:

```bash
pip install -r requirements.txt
```


### 2. Download pretrained Liveportrait weights 📥

The easiest way to download the pretrained weights is from HuggingFace:
```bash
# !pip install -U "huggingface_hub[cli]"
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```

If you cannot access to Huggingface, you can use [hf-mirror](https://hf-mirror.com/) to download:
```bash
# !pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```

Alternatively, you can download all pretrained weights from [Google Drive](https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib) or [Baidu Yun](https://pan.baidu.com/s/1MGctWmNla_vZxDbEp2Dtzw?pwd=z5cn). Unzip and place them in `./pretrained_weights`.


### 3. Download Audio Encoder Weights (Whisper)
First, download the Whisper Tiny weights file: [Whisper](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)

Then, create a subdirectory named `audio_processor` under the `./pretrained_weights` folder and place the weight file inside it (`./pretrained_weights/audio_processor`).

Ensure the directory structure matches what's shown here: [**This Repository**](assets/docs/directory-structure.md)。


### 4. Train the Model
The dataset used in this project is MEAD, a collection of facial video recordings from 60 actors performing dialogues under eight different emotions (excluding neutral) at three intensity levels. For more information, please refer to the [MEAD dataset page](https://wywu.github.io/projects/MEAD/MEAD.html).

After downloading the dataset, process it using the `process_data_MEAD.py` script:
```bash
python process_data_MEAD.py  \
  --dataset_path your_path/MEAD  \        # Path to raw MEAD data
  --output_path your_path/processed_MEAD  # Output path for processed data
```

You can then start training:
```bash
python train.py -dataset_path your_path/processed_MEAD --work_dir output
```
For detailed configuration options, refer to the training parameter file: [config](src/config/train_config.py)

### 5. Inference

Use the inference script to load the pretrained model and easily generate facial animations driven by audio.

You need to specify the `--statistic_path` to define the mean and variance of the data distribution used during prediction. If you processed the dataset using `process_data_MEAD.py`, the corresponding statistic file will be located at `.../processed_MEAD/statistic.pt`.

Also, specify the `--pretrained_model_path` to load the trained model.

Other parameters:

- `-s`: Path to the reference face image
- `-d`: Path to the driving audio file
- `-o`: Output directory for generated results
- `seed`: Random seed (optional, default is 0)
- `cfg_scale`: Guidance scale (optional, default is 4.5)
- `output_fps`: Output video frame rate (optional, default is 30)

```bash
python inference_with_audio.py \
  -s assets/examples/source/s9.jpg \
  -d assets/examples/driving/angry.m4a \
  -o generated \
  --statistic_path your_statistic_path \
  --pretrained_model_path your_model_path
```


## Acknowledgments
We would like to express our gratitude to [LivePortrait](https://github.com/KwaiVGI/LivePortrait), [Whisper](https://github.com/openai/whisper), and [MMAudio](https://github.com/hkchengrex/MMAudio) for their open-source research and contributions.
