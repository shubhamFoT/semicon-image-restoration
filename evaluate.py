import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from model import NAFNetMVP

def load_image(file_path):
    if file_path.endswith('.npy'):
        arr = np.load(file_path)
        tensor = torch.from_numpy(arr).float()
    else:
        img = Image.open(file_path).convert('L')
        tensor = ToTensor()(img)
    
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor

def main():
    parser = argparse.ArgumentParser(description="Evaluate NAFNet")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing test images")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save restored images")
    parser.add_argument('--model_path', type=str, default='best_model.pth', help="Path to trained weights")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NAFNetMVP().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    valid_exts = ('.npy', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
    input_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_exts)]
    
    if not input_files:
        print(f"No valid image files found in {args.input_dir}")
        return

    print(f"Found {len(input_files)} files. Starting evaluation...")

    with torch.no_grad():
        for filename in input_files:
            file_path = os.path.join(args.input_dir, filename)
            noisy_tensor = load_image(file_path).to(device)

            restored_tensor = model(noisy_tensor)
            restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)
            
            restored_arr = restored_tensor.squeeze().cpu().numpy()
            restored_arr = np.clip(restored_arr * 255.0, 0, 255).astype(np.uint8)
            save_path = os.path.join(args.output_dir, os.path.splitext(filename)[0] + '.png')
            Image.fromarray(restored_arr).save(save_path)

if __name__ == '__main__':
    main()