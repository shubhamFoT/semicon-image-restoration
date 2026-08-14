# 🔬 SEMICON India 2026: Edge-Ready Semiconductor Image Restoration


This repository contains an end-to-end, hardware-software co-designed deep learning pipeline for restoring noisy semiconductor chip array images. Developed for the **SEMICON India Hackathon 2026**, this solution recovers microscopic geometric structures from degraded sensor data, performs 2x Super-Resolution, and is aggressively optimized for real-time edge ASIC/FPGA deployment.

## ⚡ Advanced Edge Optimization & Hardware-Software Co-Design

1. **Multi-Objective FFT Loss Optimization:** Bypasses the traditional "blurry average" trap of spatial losses by penalizing high-frequency spectrum errors using a 2D Fast Fourier Transform (L_FFT), locking in sharp semiconductor grid geometries.
2. **INT8 Post-Training Quantization (PTQ):** Compresses the raw FP32 NAFNet PyTorch model into an ultra-low-footprint 8-bit integer ONNX graph (`nafnet_int8_quantized.onnx`) tailored for edge deployment.
3. **AXI4-Stream SimpleGate RTL Coprocessor:** A custom Verilog implementation of NAFNet's non-linear core activation (`simple_gate.v`), utilizing industry-standard AXI4-Stream bus protocols with a fully verified self-checking testbench (`tb_simple_gate.v`).
4. **SRAM-Constrained Tiled Inference (`tiled_inference.py`):** Implements a sliding-window overlap algorithm ($64 \times 64$ patches) with dynamic scale detection, overcoming off-chip memory bandwidth limitations and L1/L2 cache ceilings on edge chips.
5. **Physics-Informed Noise Modeling (`sem_noise_model.py`):** Replaces artificial Gaussian noise with a Poisson-Gaussian mixture distribution (Y ~ Poisson(λX)/λ + Gaussian(0, σ²)) to simulate true electron/photon shot noise inherent to Scanning Electron Microscopes (SEM).

## 🛠️ Production Roadmap & Hardware Constraints (Architectural Notes)
*To address production tape-out readiness, the current rapid-prototyping architecture acknowledges the following physical realities for future ASIC iteration:*
* **Memory Bandwidth & Ping-Pong Buffering:** Current tiled execution relies on host-managed streaming. Full silicon implementation requires an on-chip DMA controller and double-buffered SRAM blocks to hide memory latency.
* **Division/Square Root Hardware Scaling:** Layer Normalization layers utilize online division. Production deployment requires hardware-friendly fixed-point scaling approximations to minimize DSP slice consumption.

## 📂 Repository Structure
```text
├── hardware_acceleration/
│   ├── simple_gate.v             # AXI4-Stream Verilog DSP block for NAFNet SimpleGate
│   ├── tb_simple_gate.v          # Self-checking Icarus Verilog testbench
│   └── README.md                 # Hardware coprocessor documentation
├── dataset.py                    # Data loading and paired geometric augmentations
├── model.py                      # NAFNet architecture implementation
├── sem_noise_model.py            # Physics-informed Poisson-Gaussian noise injector
├── train.py                      # Training loop (Charbonnier + SSIM + FFT loss)
├── tiled_inference.py            # SRAM-friendly dynamic scaling & tiling logic
├── evaluate.py                   # Standalone inference script integrating tiled processing
├── export.py                     # Script to generate Edge-ready ONNX graph
├── quantize.py                   # INT8 Post-Training Quantization script
├── nafnet_int8_quantized.onnx    # Compressed, edge-ready hardware graph
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation

## 🚀 Quick Start & Reproduction

### 1. Environment Setup
```bash
git clone [https://github.com/shubhamFOT/semicon-image-restoration.git](https://github.com/shubhamFOT/semicon-image-restoration.git)
cd semicon-image-restoration
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
