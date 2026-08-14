import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    model_fp32 = 'nafnet_edge_ready.onnx'
    model_quant = 'nafnet_int8_quantized.onnx'

    print(f"Quantizing {model_fp32} to INT8...")

    # Dynamically quantize the model weights to 8-bit integers
    quantize_dynamic(
        model_input=model_fp32,
        model_output=model_quant,
        weight_type=QuantType.QUInt8
    )

    print(f"Success! Highly compressed hardware model saved to {model_quant}")

if __name__ == '__main__':
    main()
