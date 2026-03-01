# Task: Algorithm Feasibility Analysis for SoC

- [x] Research Bluetrum SoC & HiFi 5 DSP specs
- [x] Analyze algorithm complexity (FLOPs/MIPS)
- [x] Evaluate real-time feasibility
- [x] Recommend optimized alternatives for DSP

# Task: Design HiFi-5 Optimized Demucs (TinyDemucs)

- [x] Define DSP-friendly architecture (Tiny U-Net)
- [x] Implement Python simulation with block-based processing
- [x] Process `Nunchucks_jay.wav` and output to `Demucs_HiFi5`
- [x] Document C/C++ porting guidelines

# Task: Model Training via Knowledge Distillation
- [x] Draft Knowledge Distillation script (`train_kd.py`)
- [x] Document training workflow and dataset preparation
- [x] Provide instructions for integrating trained weights (`.pth`) into DSP inference

# Task: Execute KD Training and Inference
- [x] Modify `train_kd.py` to accept multiple files or a directory.
- [x] Train TinyDemucs on the 3 songs in `Audio sample/`.
- [x] Create output directory `Algorithm/Audio output/Demucs_HiFi5_output`.
- [x] Execute inference on `Nunchucks_jay.wav` using the trained weights.

# Task: Enhance TinyDemucs Architecture & Retrain
- [x] Upgrade U-Net depth (4 layers), channels (up to 256), and add Dilated Convolutions.
- [x] Increase epoch count and dataset sampling rate for Overfitting test.
- [x] Retrain the model on the `Audio sample/` directory.
- [x] Run inference again on `Nunchucks_jay.wav`.

# Task: Analysis and SoC Deployment Roadmap
- [x] Analyze causes of noise in the KD output.
- [x] Define steps for dataset scaling and architecture refinement.
- [x] Outline the C/C++ (INT16) porting process for HiFi 5 DSP.

# Task: GitHub Repository Backup
- [/] Initialize local git repository, commit files, and configure `.gitignore`.
- [ ] Push local repository to GitHub (`ginoliaotw/TinyDemucs-HiFi5`).
# Task: Multi-Model Voice Separation

- [x] Set up Demucs
    - [x] Install Demucs
    - [x] Install dependencies
    - [x] Run inference on Nunchucks_jay.wav
    - [x] Record execution time (~13.5s)
- [x] Set up Audio-Separator (UVR Backend)
    - [x] Install Audio-Separator
    - [x] Install dependencies
    - [x] Run inference on Nunchucks_jay.wav
    - [x] Record execution time (~39.2s)
- [x] Set up Vocal-Remover (UNet v5)
    - [x] Clone repository
    - [x] Install dependencies
    - [x] Run inference on Nunchucks_jay.wav
    - [x] Record execution time (~35.3s)
- [x] Verify all output files and summarize times
