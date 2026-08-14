# Hardware Acceleration: SSIM Coprocessor

To make this image restoration pipeline viable for real-time edge deployment (e.g., inline semiconductor inspection), we must optimize the computational bottlenecks. 

Evaluating the Structural Similarity Index Measure (SSIM) across thousands of sliding image windows is highly taxing on a standard CPU/GPU due to the localized variance and covariance formulas. 

### Pipelined Difference & Squaring Unit (`ssim_diff_squarer.v`)
This Verilog RTL module demonstrates a hardware-software co-design approach. By offloading the (x - y)^2 operations to a dedicated 2-stage pipelined DSP block on an FPGA, we can process pixel streams at high clock frequencies with a throughput of one squared difference per clock cycle, drastically reducing the SSIM evaluation latency.
