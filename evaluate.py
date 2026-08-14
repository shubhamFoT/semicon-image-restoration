import os
import argparse
import torch
import numpy as np
from model import NAFNetMVP

def evaluate(input_dir, output_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")
    
    model = NAFNetMVP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for img_name in os.listdir(input_dir):
            if not img_name.endswith('.npy'):
                continue
                
            img_path = os.path.join(input_dir, img_name)
            noisy_arr = np.load(img_path)
            noisy_tensor = torch.from_numpy(noisy_arr).float()
            
            if noisy_tensor.ndim == 2:
                noisy_tensor = noisy_tensor.unsqueeze(0)
                
            noisy_tensor = noisy_tensor.unsqueeze(0).to(device)
            restored_tensor = model(noisy_tensor)
            
            out_arr = restored_tensor.squeeze().cpu().numpy()
            np.save(os.path.join(output_dir, img_name), out_arr)
            
    print(f"Restoration complete. Arrays saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image restoration inference.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing noisy test arrays.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save restored arrays.")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to the trained model weights.")
    
    args = parser.parse_args()
    evaluate(args.input_dir, args.output_dir, args.model_path)
