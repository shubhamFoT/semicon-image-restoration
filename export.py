import torch
from model import NAFNetMVP

# 1. Initialize the model and load your trained weights
model = NAFNetMVP()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# 2. Create a dummy tensor matching your input dimensions (1 batch, 1 channel, 256x256)
dummy_input = torch.randn(1, 1, 256, 256) 

# 3. Export the model to ONNX format
torch.onnx.export(
    model, 
    dummy_input, 
    "nafnet_edge_ready.onnx",
    export_params=True,
    opset_version=11,          # Standard, stable opset version
    do_constant_folding=True,  # Optimizes the graph for inference
    input_names=['input'],
    output_names=['output']
)
print("Model successfully exported to nafnet_edge_ready.onnx")
