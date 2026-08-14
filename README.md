# 🔬 SEMICON India 2026: Edge-Ready Semiconductor Image Restoration

This repository contains an end-to-end deep learning pipeline for restoring noisy semiconductor chip array images. Developed for the **SEMICON India Hackathon 2026**, this solution not only recovers microscopic geometric structures from degraded sensor data but is actively optimized for real-time edge deployment.

## ✨ Key Features
- **Architecture:** NAFNet-inspired model utilizing `SimpleGate` non-linear activations to efficiently capture high-frequency spatial details.
- **Advanced Loss Optimization:** A custom loss function combining **Charbonnier Loss** (robust edge preservation) and **SSIM Loss** (structural geometry enforcement) to prevent blurring.
- **Data Augmentation:** Generalized against spatial variations using strictly paired geometric transformations (rotations and flips) to prevent overfitting.
- **Edge Deployment Ready:** Automated `ONNX` graph generation with separated data weights for immediate deployment to inline inspection hardware.
- **Hardware-Software Co-Design:** Includes custom Verilog RTL for a 2-stage pipelined SSIM Difference & Squaring unit, demonstrating how to offload evaluation bottlenecks to an FPGA.

## 📂 Repository Structure
```text
├── hardware_acceleration/
│   ├── ssim_diff_squarer.v   # Pipelined Verilog DSP block for SSIM computation
│   └── README.md             # Hardware coprocessor documentation
├── dataset.py                # Data loading and paired geometric augmentations
├── model.py                  # NAFNet architecture implementation
├── train.py                  # Training loop (Charbonnier + SSIM loss)
├── evaluate.py               # Standalone inference script for judges
├── export.py                 # Script to generate Edge-ready ONNX graph
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
