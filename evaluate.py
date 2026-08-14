import os
import argparse
import torch
import numpy as np
from model import NAFNetMVP
from tiled_inference import compute_tiled_inference

def main():
    parser = argparse.ArgumentParser(description="Evaluate NAFNet with SRAM-Constrained Tiled Inference")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing noisy .npy files")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save restored .npy files")
    parser.add_argument('--model_path', type=str, default='best_model.pth', help="Path to trained weights")
    parser.add_argument('--tile_size', type=int, default=64, help="Size of the SRAM-friendly tiles")
    parser.add_argument('--overlap', type=int, default=8, help="Overlap between tiles to prevent seams")
    args = parser.parse_args()

    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on: {device}")

    # 2. Load Model
    model = NAFNetMVP().to(device)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found at {args.model_path}")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 3. Setup Output Directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Process Images
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith('.npy')]
    if not input_files:
        print(f"No .npy files found in {args.input_dir}")
        return

    print(f"Found {len(input_files)} files. Starting SRAM-constrained tiled inference...")

    with torch.no_grad():
        for filename in input_files:
            file_path = os.path.join(args.input_dir, filename)
            noisy_arr = np.load(file_path)
            
            # Convert to tensor [Batch, Channel, Height, Width]
            noisy_tensor = torch.from_numpy(noisy_arr).float().to(device)
            if noisy_tensor.ndim == 2:
                noisy_tensor = noisy_tensor.unsqueeze(0).unsqueeze(0)
            elif noisy_tensor.ndim == 3:
                noisy_tensor = noisy_tensor.unsqueeze(0)

            # --- TILED INFERENCE ---
            restored_tensor = compute_tiled_inference(
                model, 
                noisy_tensor, 
                tile_size=args.tile_size, 
                overlap=args.overlap
            )

            # Convert back to numpy and save
            restored_arr = restored_tensor.squeeze().cpu().numpy()
            save_path = os.path.join(args.output_dir, filename)
            np.save(save_path, restored_arr)

            print(f"Restored and saved: {filename}")
            
    print("Evaluation complete. All files processed successfully using tiled memory management.")

if __name__ == '__main__':
    main()
