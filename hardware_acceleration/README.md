# 🎛️ Hardware Acceleration & RTL IP Block

This directory contains the Verilog hardware descriptions for the edge-accelerated NAFNet coprocessor pipeline.

## 📂 File Manifest
* **`simple_gate.v`**: AXI4-Stream compliant hardware implementation of the NAFNet `SimpleGate` activation function. Replaces expensive software exponentiations with single-cycle parallel integer multiplication.
* **`tb_simple_gate.v`**: Self-checking functional testbench validating dual-stream INT8 vector inputs against clock cycles, backpressure states, and assertions using Icarus Verilog (`iverilog`).

## 🚀 Simulation & Verification
To run the self-checking hardware testbench locally via CLI:
```bash
iverilog -o sim tb_simple_gate.v simple_gate.v
vvp sim