# Implementation Plan - Multi-Model Voice Separation

Install and run three state-of-the-art voice separation models on `Nunchucks_jay.wav`.

## Proposed Changes

### 1. Demucs
- Directory: `/Users/ginoliaotpv/Antigravity/Algorithm/Demucs`
- Goal: Run standard Demucs 4-stem separation.

### 2. Audio-Separator (UVR-CLI)
- Directory: `/Users/ginoliaotpv/Antigravity/Algorithm/AudioSeparator`
- Goal: Use a high-quality MDX-Net or RoFormer model via the UVR backend.

### 3. Vocal-Remover
- Directory: `/Users/ginoliaotpv/Antigravity/Algorithm/VocalRemover`
- Goal: Run SOTA RoFormer-based separation.

## Output Naming Convention
- `Nunchucks_jay_Demucs_output.wav`
- `Nunchucks_jay_UVR_output.wav`
- `Nunchucks_jay_RoFormer_output.wav`

## Verification Plan
- Ensure all 3 output files are present in `/Users/ginoliaotpv/Antigravity/Algorithm/Audio sample/`.

---

## Feasibility Report: Bluetrum SoC (HiFi 5) Deployment

### 1. Hardware Constraints (HiFi 5 DSP)
- **Peak Compute**: ~8 GFLOPS (SP Floating Point) / ~32-64 GOPS (INT8 NN Engine).
- **RAM**: Typically < 1MB available for buffers (Bluetrum specific context).
- **Clock**: ~270MHz - 500MHz.

### 2. Algorithm Comparison (Real-Time Suitability)

| Algorithm | FLOPs (per sec) | Model Size | Real-Time Feasibility |
| :--- | :--- | :--- | :--- |
| **HTDemucs** | ~2,000 GFLOPs | ~160 MB | **Infeasible** (Too heavy) |
| **MDX-Net** | ~50-100 GFLOPs | ~50-80 MB | **Infeasible** (Memory/Compute) |
| **Edge-BS-RoFormer** | ~11 GFLOPs | ~8.5 MB | **Borderline** (Requires INT8 Quant) |
| **Conv-TasNet (Lite)**| **< 1 GFLOP** | **< 1 MB** | **Ideal** (DSP optimized) |

### 3. Conclusion & Recommendations
The current SOTA models tested (Demucs, MDX-Net) are **strictly too intensive** for an embedded SoC like Bluetrum to run in real-time.

**To achieve real-time on HiFi 5:**
1. **Architecture**: Switch to Lightweight Time-Domain models (e.g., **PoCoNet** or **Conv-TasNet**).
2. **Quantization**: Convert model to **INT8/INT16** using Cadence's XTENSA NN library.
3. **Memory Management**: Use ping-pong buffers for STFT/iSTFT and optimize the weights for Flash execution (XIP) if memory is tight.

---

## HiFi-5 Optimized Prototype (TinyDemucs)

To transition from the high-level Python environment to the Bluetrum HiFi 5 DSP, I will design a strictly constrained, streaming-friendly architecture called **TinyDemucs**.

### Architecture Design principles
1. **Block-based Processing (Streaming)**: Audio is processed in small buffers (e.g., 2048 samples = ~46ms at 44.1kHz). This eliminates the need to load the entire song into RAM, simulating DMA ping-pong buffers.
2. **1D Convolutions Only**: Avoided 2D convolutions or transformers. 1D Depthwise separable convolutions map perfectly to HiFi 5's MAC units.
3. **Restricted Activations**: Replaced GELU/Sigmoid with ReLU, which operates natively on fixed-point integers without LUTs (Look-Up Tables).
4. **Quantization Readiness**: Operations are structured to easily allow `int16` activations and `int8` weights in the C/C++ port using the Cadence NatureDSP library.

### Guidelines for Future C/C++ Porting
When transplanting this Python logic to the Bluetrum SoC Firmware:
- **NatureDSP Library**: Use Cadence's `NatureDSP_Signal` or `NatureDSP_Math` libraries to perform the 1D convolution dot products.
- **Fixed-Point Quantization**: The model must undergo Quantization-Aware Training (QAT). The C++ code should define `int16_t` arrays for activations and `int8_t` for weights.
- **State/History Buffers**: In Python `tinydemucs_dsp.py`, we pad the input. In C++, you must implement a "Delay Line" ring-buffer to carry over the `kernel_size - 1` samples from the previous DMA block into the current convolution.

---

## Model Training (Knowledge Distillation)

Since TinyDemucs is built from scratch for hardware constraints, it starts with untrained (random) weights. It must be trained using **Knowledge Distillation (KD)**, treating the original heavy Demucs model as the "Teacher".

### 1. Training Script (`train_kd.py`)
I will provide a PyTorch script (`train_kd.py` in `Demucs_HiFi5/`) that performs the following loop:
1. Loads an original audio mix (Input).
2. Runs the heavy Demucs (Teacher) to generate the "Perfect Vocals" (Ground Truth).
3. Runs the TinyDemucs (Student) to generate its attempted "Vocals".
4. Calculates the Loss (L1/MSE difference) between Student and Teacher.
5. Backpropagates to update TinyDemucs weights.

### 2. Integration into DSP Algorithm
Once trained, the script will output `tinydemucs_best.pth`. This file will be loaded into the exact structural definition within `tinydemucs_dsp.py` via `model.load_state_dict()`, replacing the random weights with highly optimized, hardware-ready separation filters.

### Execution Plan
- Create directory `/Users/ginoliaotpv/Antigravity/Algorithm/Demucs_HiFi5`.
- Write `tinydemucs_dsp.py` containing the PyTorch network and the simulated C++ inference loop.
- Process `../Audio sample/Nunchucks_jay.wav` sequentially.
- Save output as `Nunchucks_jay_TinyDemucs_output.wav`.

---

## Strategic Blueprint: Scaling for Production (Bluetrum HiFi 5)

When transitioning from the Python prototype to a commercial product using your 10,000-song dataset, the development strategy is critical.

### Why Direct Pruning of Demucs Fails (Approach 1)
Taking the original `htdemucs` architecture and attempting to "shrink" it (Pruning/Compression) to fit the HiFi 5 is **highly discouraged**.
- **Architectural Mismatch**: Demucs uses complex ops (BiLSTMs, Transformers, GELU) that do not map efficiently to the HiFi 5 DSP's VLIW pipeline.
- **Memory Walls**: Even heavily pruned, the layer activation sizes of a standard U-Net will instantly overflow the Bluetrum SoC's limited SRAM.

### The Recommended Workflow (Approach 2: Hardware-Aware KD)
The optimal path is **Hardware-Aware Architecture Design + Knowledge Distillation**, exactly as prototyped in `TinyDemucs`, but scaled up:

1.  **Define the Hardware Limits First (The "Box")**:
    *   Work with your Firmware engineers to define exact SoC limits: *Maximum SRAM per block* and *Maximum MACs per cycle*.
    *   Design a PyTorch architecture (like `TinyDemucs` or a `Conv-TasNet` variant) that fits **strictly** inside this box. Use only 1D Convolutions and ReLUs.
2.  **Generate Soft Labels (The "Teacher")**:
    *   Run your 10,000-song dataset through the heavy, original **Demucs** model on a GPU server.
    *   Save the high-quality separated vocals. This becomes your "Ground Truth / Soft Labels".
3.  **Knowledge Distillation (The "Student")**:
    *   Train your custom, hardware-friendly PyTorch model to mimic the Demucs outputs, using the 10,000-song dataset.
    *   *Why this works*: It is mathematically easier for a small network to learn the "smoothed" feature mappings produced by a Teacher network than to learn from raw, noisy human stems.
4.  **Quantization-Aware Training (QAT，關鍵步驟)**:
    *   In the final 10% of training epochs, enable PyTorch QAT.
    *   This forces the model to simulate `INT16` (Activations) and `INT8` (Weights) precision loss, ensuring the audio doesn't degrade when ported to the HiFi 5.
5.  **C/C++ DSP Porting**:
    *   Export the QAT weights.
    *   Rewrite the inference loop in C++ using Cadence `NatureDSP` libraries, managing your own delay lines for the convolutions.
