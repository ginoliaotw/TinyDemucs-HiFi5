# TinyDemucs for HiFi 5 DSP

This repository contains the PyTorch prototype and Knowledge Distillation (KD) training workflow for **TinyDemucs**, an ultra-lightweight, 1D convolution-based audio separation architecture designed specifically to fit the constraints of the Bluetrum HiFi 5 SoC.

## Project Context
The full architectural blueprint, evaluation reports, and C/C++ DSP porting roadmap are located in the `docs/` directory.

---

## 🚀 Quick Start Guide: Windows 10 (RTX Series GPU)

If you have just cloned this repository onto a Windows 10 machine equipped with an NVIDIA RTX GPU (e.g., RTX 5090), follow these steps to resume development and launch large-scale training.

### Step 1: Tell Antigravity to Read the Docs
If you are using the AI assistant **Antigravity** on this new machine, open this project folder via VS Code/Terminal, and give it the following initial prompt:
> **"請閱讀 `docs/` 目錄下的文件 (`walkthrough.md`, `implementation_plan.md`)，並幫我準備好 Windows 10 RTX 5090 的開發環境與後續的大規模訓練。"**

Antigravity will instantly inherit all architectural context, past bug fixes, and the development roadmap.

### Step 2: Set up Python Environment (Windows)
Open PowerShell or Command Prompt in this repository directory:
```powershell
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
.\venv\Scripts\activate

# 3. Install requirements (PyTorch with CUDA support is recommended)
pip install -r requirements.txt
# Note: For RTX 5090, you may need a specific PyTorch index URL for CUDA 12.x.
# pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Prepare the Dataset
Place your large dataset (e.g., 10,000 `.wav`/`.mp3` songs) into a directory on your Windows machine, for example:
`C:\Data\Audio_Dataset\`

### Step 4: Launch Massive Training (Knowledge Distillation)
The training script will automatically detect the NVIDIA GPU (`cuda`) and utilize the massive Tensor Cores of the RTX 5090.
Run the training hook with your desired parameters (e.g., 1000 epochs):
```powershell
python train_kd.py -i "C:\Data\Audio_Dataset" -e 1000 -s 1000 -o .
```

### Step 5: Test the Inference
After training, run the DSP simulation to hear the output on your Windows machine:
```powershell
python tinydemucs_dsp.py -i "C:\Data\Audio_Dataset\Test_Song.wav" -o "Demo_Output.wav" -w tinydemucs_best.pth
```
