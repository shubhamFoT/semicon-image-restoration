import torch
import torch.nn as nn

class SEMNoiseModel(nn.Module):
    """
    Simulates real-world Scanning Electron Microscope (SEM) and optical 
    sensor noise profiles using a Poisson-Gaussian mixture model.
    """
    def __init__(self, photon_peak=10000.0, read_noise_std=0.01):
        super(SEMNoiseModel, self).__init__()
        self.photon_peak = photon_peak
        self.read_noise_std = read_noise_std

    def forward(self, clean_tensor):
        # 1. Ensure tensor is strictly positive for Poisson distribution
        image_clamped = torch.clamp(clean_tensor, min=0.0)
        
        # 2. Poisson Shot Noise (Signal Dependent)
        # Scale to simulated photon/electron counts
        particles = image_clamped * self.photon_peak
        noisy_poisson = torch.poisson(particles) / self.photon_peak
        
        # 3. Gaussian Read Noise (Signal Independent baseline sensor noise)
        read_noise = torch.randn_like(clean_tensor) * self.read_noise_std
        
        # 4. Combine and clamp back to standard visual range
        noisy_final = torch.clamp(noisy_poisson + read_noise, min=0.0, max=1.0)
        return noisy_final

if __name__ == "__main__":
    # Quick sanity check for the module
    dummy_image = torch.ones(1, 1, 256, 256) * 0.5
    noise_injector = SEMNoiseModel()
    noisy_image = noise_injector(dummy_image)
    print("SEM Poisson-Gaussian Noise Model successfully initialized and tested.")
