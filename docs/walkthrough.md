## Multi-Model Voice Separation (Karaoke Optimization)

I have installed and executed the top 3 separation models to compare their performance and quality.

### Execution Results Summary

| Model / Algorithm | Output File | Execution Time | Notes |
| :--- | :--- | :--- | :--- |
| **Demucs (htdemucs)** | [Nunchucks_jay_Demucs_output.wav](file:///Users/ginoliaotpv/Antigravity/Algorithm/Audio%20sample/Nunchucks_jay_Demucs_output.wav) | **13.5s** | Fastest, balanced quality. |
| **Vocal-Remover (U-Net v5)** | [Nunchucks_jay_RoFormer_output.wav](file:///Users/ginoliaotpv/Antigravity/Algorithm/Audio%20sample/Nunchucks_jay_RoFormer_output.wav) | **35.3s** | Classic high-quality U-Net. |
| **Audio-Separator (MDX-Net)** | [Nunchucks_jay_UVR_output.wav](file:///Users/ginoliaotpv/Antigravity/Algorithm/Audio%20sample/Nunchucks_jay_UVR_output.wav) | **39.2s** | UVR backend, very clean vocals. |

### Installation Details
Each model was installed in its own directory within `Algorithm/`:
- [`Demucs/`](file:///Users/ginoliaotpv/Antigravity/Algorithm/Demucs/)
- [`AudioSeparator/`](file:///Users/ginoliaotpv/Antigravity/Algorithm/AudioSeparator/) (Note: Installed in Demucs venv to save space/time)
- [`VocalRemover/`](file:///Users/ginoliaotpv/Antigravity/Algorithm/VocalRemover/)

### Recommended for Karaoke
1.  **Audio-Separator (UVR)**: Offers the cleanest vocal removal for backing tracks.
2.  **Demucs**: Best if you also need to isolate drums/bass for a full remix.

---

## Bluetrum HiFi-5 DSP Optimization (TinyDemucs)

To transition from the high-level Python environment to the Bluetrum HiFi 5 DSP, I designed a constrained, streaming-friendly architecture called **TinyDemucs** located in [`Algorithm/Demucs_HiFi5`](file:///Users/ginoliaotpv/Antigravity/Algorithm/Demucs_HiFi5/).

### Prototype Architecture (`tinydemucs_dsp.py`)
This script simulates how the algorithm will run in C++ on the SoC:
1. **Block-based Streaming (Ping-Pong Simulation)**: Instead of processing the whole file, audio is read in chunks of `2048` samples (~46ms), simulating a hardware DMA cycle.
2. **1D Depthwise Convolutions**: All 2D/Transformer ops were replaced with `Conv1d(groups=channels)` to maximize the MAC efficiency on the HiFi 5.
3. **Restricted Non-Linearities**: Used `ReLU` exclusively because it maps natively to standard INT16 conditional instructions, avoiding the expensive Look-Up Tables (LUT) required for GELU or Sigmoid.

### Guidelines for Future C/C++ Porting
When transplanting this Python logic to the Bluetrum SoC Firmware:
- **NatureDSP Library**: Use Cadence's `NatureDSP_Signal` or `NatureDSP_Math` libraries to perform the 1D convolution dot products.
- **Fixed-Point Quantization**: The model must undergo Quantization-Aware Training (QAT). The C++ code should define `int16_t` arrays for activations and `int8_t` for weights.
- **State/History Buffers**: In Python `tinydemucs_dsp.py`, we pad the input. In C++, you must implement a "Delay Line" ring-buffer to carry over the `kernel_size - 1` samples from the previous DMA block into the current convolution.

---

## Model Training: Knowledge Distillation (KD)

Because `TinyDemucs` is an entirely new, heavily constrained architecture designed specifically for the HiFi 5 DSP, it cannot use pre-existing weights. It must be trained over several days to "imitate" the original Demucs model.

### 1. Training Workflow (`train_kd.py`)
I have created a dedicated Knowledge Distillation script `Algorithm/Demucs_HiFi5/train_kd.py`.
This script acts as the bridge:
1. It loads the heavy `htdemucs` model as the **Teacher**.
2. It takes raw, mixed audio and asks the Teacher for the "Perfect Vocals".
3. It asks `TinyDemucs` (the **Student**) to generate vocals.
4. It calculates the L1 Loss difference and updates the Student's weights.

**How to run a simulation:**
```bash
cd /Users/ginoliaotpv/Antigravity/Algorithm/Demucs_HiFi5
source venv/bin/activate
python3 train_kd.py -i "../Audio sample/Nunchucks_jay.wav" -e 5 -s 10 -o .
```
*(Note: In production, the script should be modified to iterate through a large dataset like MUSDB18).*

### 2. Output and Integration
Once the training concludes, the script saves the optimized weights as `tinydemucs_best.pth`.

To run the block-based DSP inference with the newly trained weights, simply pass the `-w` flag to the DSP simulation script:

```bash
python3 tinydemucs_dsp.py -i "../Audio sample/Nunchucks_jay.wav" -o "output.wav" -w tinydemucs_best.pth
```
This entirely replaces the randomized weights with the trained filters, activating the vocal separation capabilities.

---

## Output Analysis & SoC Deployment Roadmap

While the enhanced 4-layer `TinyDemucs` architecture successfully suppresses background music, the output contains noticeable noise. This is completely expected for an edge-optimized AI model in its embryonic stage.

### 1. Analysis of the Noise Issue
- **Time-Domain Aliasing (Phase Distortion)**: We are operating directly on the raw waveform (Time-Domain) rather than the Spectrogram (Frequency-Domain) to save DSP cycles. Time-domain networks are notoriously difficult to train because mispredicting the "phase" of a wave results in audible static/white noise.
- **Micro-Dataset Underfitting**: Deep Learning models require diversity to generalize. We trained on just 3 songs. The network memorized specific waveforms but doesn't truly understand the concept of "vocals," leading to artifacts when reconstructing the wave.
- **Lack of STFT/iSTFT Smoothing**: Traditional models use Inverse Short-Time Fourier Transform, which naturally smooths out block-to-block discontinuities. Our direct 1D Conv output lacks this mathematical smoothing.

### 2. Roadmap to Bluetrum HiFi 5 Deployment
To transform this prototype into a commercial product running on the BT8951H (HiFi 4/5) SoC:

#### Phase A: Algorithm Maturation (Python)
1.  **Scale the Dataset**: The algorithm team must train `TinyDemucs` on a dataset like **MUSDB18-HQ** (150+ full tracks) for at least 500-1000 epochs.
2.  **Hybrid Architecture (Optional)**: If time-domain noise remains unacceptable, switch the first/last layer of `TinyDemucs` to a lightweight STFT/iSTFT and process the magnitudes only. This adds compute but drastically improves audio smoothness.
3.  **Loss Function Tuning**: Replace the simple `L1Loss` with a multi-resolution STFT Loss (aural loss) to force the network to prioritize human-perceivable audio quality over generic math errors.

#### Phase B: Fixed-Point Quantization (QAT)
1.  **INT16 Activation / INT8 Weights**: The HiFi 5 DSP gets a 4x MAC boost when using 8-bit or 16-bit integers. Do not use Float32 on the chip.
2.  Use PyTorch's `torch.quantization` module to perform **Quantization-Aware Training (QAT)**. This ensures the model learns to tolerate the precision loss of integer math *while* it is still training.

#### Phase C: C/C++ DSP Implementation (Firmware)
1.  **Ping-Pong DMA Buffers**: Implement the `2048` sample processing block seen in `tinydemucs_dsp.py` using hardware DMA interrupts. 
2.  **Delay Line Management**: `Conv1d` with `kernel_size=3` requires memory of the past. The C++ code must implement a ring buffer to preserve `(kernel_size - 1) * dilation` samples from the *previous* DMA interrupt block to feed into the *current* convolution block.
3.  **Cadence NatureDSP Library**: Do not write `for` loops for convolutions. Use Cadence's highly optimized assembly hooks:
    *   Initialize `NatureDSP_Signal_fir` for each convolutional layer.
    *   Ensure all data arrays are 32-byte aligned for the VLIW vector engine.
