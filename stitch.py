import os
import numpy as np
from PIL import Image, ImageDraw

# Select a sample file ID from your dataset
file_id = "000000"

# Paths to the raw noisy input and the model's restored output
noisy_path = f"./data/train/noisy/{file_id}.npy"
restored_path = f"./demo_output/{file_id}.png"

if not os.path.exists(noisy_path) or not os.path.exists(restored_path):
    print(f"Error: Could not find files for ID {file_id}. Check your paths!")
    exit()

# 1. Load the noisy input (.npy format)
noisy_arr = np.load(noisy_path)
if noisy_arr.ndim == 3:
    noisy_arr = noisy_arr.squeeze(0)  # Remove channel dim if present
noisy_img = Image.fromarray(np.clip(noisy_arr * 255.0, 0, 255).astype(np.uint8)).convert("RGB")

# 2. Load the restored output (.png format)
restored_img = Image.open(restored_path).convert("RGB")

# Ensure matching dimensions for a clean side-by-side comparison
w1, h1 = noisy_img.size
w2, h2 = restored_img.size
if h1 != h2:
    restored_img = restored_img.resize((int(w2 * (h1 / h2)), h1))
    w2, h2 = restored_img.size

# 3. Create a side-by-side canvas with space for labels
spacing = 15
canvas_width = w1 + w2 + spacing
canvas_height = h1 + 50  # Extra space at the top for title text
canvas = Image.new("RGB", (canvas_width, canvas_height), color=(245, 245, 245))

# Paste images onto canvas
canvas.paste(noisy_img, (0, 50))
canvas.paste(restored_img, (w1 + spacing, 50))

# 4. Draw clean labels
draw = ImageDraw.Draw(canvas)
draw.text((20, 15), "Before: Degraded / Noisy Input (KLA)", fill=(40, 40, 40))
draw.text((w1 + spacing + 20, 15), "After: NAFNet-Nano Restored Output", fill=(0, 100, 0))

# 5. Save directly over final_results.png
canvas.save("final_results.png")
print("Successfully generated and updated final_results.png!")