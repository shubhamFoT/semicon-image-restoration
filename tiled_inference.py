import torch

def compute_tiled_inference(model, image_tensor, tile_size=64, overlap=8):
    """
    Splits high-resolution semiconductor images into overlapping patches.
    Dynamically supports both 1x Denoising and Nx Super-Resolution by 
    auto-detecting the spatial scaling factor of the model.
    """
    _, _, h, w = image_tensor.shape
    stride = tile_size - overlap
    
    # --- AUTO-DETECT SCALE FACTOR ---
    # Run a dummy tile to see if the model is doing Super-Resolution
    dummy_tile = image_tensor[:, :, 0:tile_size, 0:tile_size]
    with torch.no_grad():
        dummy_out = model(dummy_tile)
    
    scale_h = dummy_out.shape[2] // dummy_tile.shape[2]
    scale_w = dummy_out.shape[3] // dummy_tile.shape[3]
    
    # Create output buffers scaled to match the model's output resolution
    out_h, out_w = h * scale_h, w * scale_w
    output = torch.zeros((image_tensor.shape[0], dummy_out.shape[1], out_h, out_w), device=image_tensor.device)
    weight_map = torch.zeros_like(output)
    
    # --- SLIDING WINDOW INFERENCE ---
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 1. Calculate input coordinates
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            # 2. Extract input tile
            tile = image_tensor[:, :, y_start:y_end, x_start:x_end]
            
            # 3. Process tile
            with torch.no_grad():
                processed_tile = model(tile)
            
            # 4. Calculate output coordinates (scaled)
            out_y_start = y_start * scale_h
            out_y_end = y_end * scale_h
            out_x_start = x_start * scale_w
            out_x_end = x_end * scale_w
            
            # 5. Accumulate into the expanded output buffer
            output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += processed_tile
            weight_map[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += 1.0
            
    # Average the overlapping regions to prevent visible seams
    return output / weight_map

if __name__ == "__main__":
    print("Tiled inference utility (with auto-scaling) initialized.")
