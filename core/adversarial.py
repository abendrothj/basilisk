
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from .perceptual_hash import load_video_frames, extract_perceptual_features, compute_perceptual_hash, hamming_distance

def create_gabor_kernel_torch(ksize=21, sigma=5, theta=0, lambd=10, gamma=0.5):
    """
    Creates a Gabor kernel using OpenCV and converts to PyTorch tensor.
    Matches cv2.getGaborKernel.
    """
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, np.deg2rad(theta), lambd, gamma)
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # Repeat for 3 channels (groups=3)
    # Shape: (3, 1, H, W)
    kernel_tensor = kernel_tensor.repeat(3, 1, 1, 1)
    return kernel_tensor

def soft_histogram(values, bins=32, min_val=0, max_val=256, sigma=4.0):
    """
    Differentiable soft histogram.
    values: (B, C, H, W)
    Returns: (B, C * bins) flattened
    """
    if values.dim() == 4:
        values = values.flatten(2) # (B, C, H*W)
    
    bin_centers = torch.linspace(min_val, max_val, bins, device=values.device)
    values_exp = values.unsqueeze(-1)
    centers_exp = bin_centers.view(1, 1, 1, bins)
    
    dists = (values_exp - centers_exp) ** 2
    weights = torch.exp(-dists / (2 * sigma**2))
    hist = weights.sum(dim=2) # (B, C, bins)
    
    return hist.flatten(start_dim=1)

def extract_features_differentiable(frames_torch):
    """
    Differentiable approximation of perceptual features.
    Args:
        frames_torch: (T, 3, H, W) tensor, values 0-255 (float32)
    Returns:
        flat_features: (T, Feature_Size) tensor
    """
    T, C, H, W = frames_torch.shape
    
    # 1. Edges (Sobel approximation of Canny)
    gray = 0.299 * frames_torch[:, 2, :, :] + 0.587 * frames_torch[:, 1, :, :] + 0.114 * frames_torch[:, 0, :, :]
    gray = gray.unsqueeze(1)
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=frames_torch.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=frames_torch.device).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    
    magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
    # Soft thresholdting (sigmoid around 150)
    edges = torch.sigmoid((magnitude - 150) / 10.0) * 255.0
    edges_flat = edges.view(T, -1)
    
    # 2. Textures (Gabor)
    textures_list = []
    for theta in [0, 45, 90, 135]:
        kernel = create_gabor_kernel_torch(theta=theta).to(frames_torch.device)
        out = F.conv2d(frames_torch, kernel, padding=10, groups=3)
        textures_list.append(out)
        
    # Stack and flatten exactly like core
    # Core: [0, 45, 90, 135] -> Stack -> (4, H, W, 3) -> Flatten
    # Here: List order is correct.
    textures_stacked = torch.stack(textures_list, dim=1) # (T, 4, 3, H, W)
    textures_permuted = textures_stacked.permute(0, 1, 3, 4, 2) # (T, 4, H, W, 3)
    textures_flat = textures_permuted.reshape(T, -1)
    
    # 3. Saliency (Laplacian)
    laplacian_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=frames_torch.device).view(1, 1, 3, 3)
    laplacian_k = laplacian_k.repeat(3, 1, 1, 1)
    saliency = F.conv2d(frames_torch, laplacian_k, padding=1, groups=3)
    saliency_permuted = saliency.permute(0, 2, 3, 1) # (T, H, W, 3)
    saliency_flat = saliency_permuted.reshape(T, -1)
    
    # 4. Color Hist
    hist_all = soft_histogram(frames_torch, bins=32)
    # hist_all is [Blue, Green, Red] (channel 0, 1, 2)
    # We use it directly as it matches Core's extraction order logic (by accident or design)
    hist_flat = hist_all
    
    flat_features = torch.cat([edges_flat, textures_flat, saliency_flat, hist_flat], dim=1)
    return flat_features

def compute_hash_differentiable(frames_torch, projection_matrix):
    """
    Computes soft hash values (logits).
    """
    features = extract_features_differentiable(frames_torch)
    norms = torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8
    features_norm = features / norms
    projected = features_norm @ projection_matrix
    projected_mean = torch.mean(projected, dim=0)
    return projected_mean

def generate_projection_matrix(frame_shape):
    """
    Generates the random projection matrix matching core/perceptual_hash.py
    """
    H, W, C = frame_shape
    feature_len = (H * W) + (4 * H * W * 3) + (H * W * 3) + 96
    hash_size = 256
    
    # Must match the seed in perceptual_hash.py exactly
    np.random.seed(42)
    projection = np.random.randn(feature_len, hash_size)
    return torch.tensor(projection, dtype=torch.float32)

def poison_video(
    video_path, 
    target_hash_bits, 
    output_path, 
    epsilon=0.1, 
    num_iterations=40, 
    learning_rate=2.0,
    verbose=False
):
    """
    Main entry point for poisoning a video.
    """
    original_frames_list = load_video_frames(video_path, max_frames=30)
    if not original_frames_list:
        raise ValueError(f"Could not load video: {video_path}")
        
    H, W, C = original_frames_list[0].shape
    
    # Prepare Torch Tensors
    original_np = np.stack(original_frames_list)
    original_np = np.transpose(original_np, (0, 3, 1, 2)) # (T, C, H, W)
    
    frames_torch = torch.tensor(original_np, dtype=torch.float32).clone().detach()
    frames_torch.requires_grad = True
    
    target_hash_torch = torch.tensor(target_hash_bits, dtype=torch.float32)
    projection_matrix = generate_projection_matrix((H, W, C))
    
    optimizer = torch.optim.Adam([frames_torch], lr=learning_rate)
    clean_frames = torch.tensor(original_np, dtype=torch.float32)
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        proj_mean = compute_hash_differentiable(frames_torch, projection_matrix)
        logits = proj_mean - proj_mean.mean()
        
        loss = F.binary_cross_entropy_with_logits(logits, target_hash_torch)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            diff = frames_torch - clean_frames
            diff = torch.clamp(diff, -epsilon * 255, epsilon * 255)
            frames_torch.data = clean_frames + diff
            frames_torch.data = torch.clamp(frames_torch.data, 0, 255)
            
        if verbose and (i+1) % 10 == 0:
            print(f"Iter {i+1}: Loss {loss.item():.4f}")
            
    # Save Result
    poisoned_np = frames_torch.detach().numpy()
    frames_list = [
        np.transpose(frame, (1, 2, 0)).astype(np.uint8) 
        for frame in poisoned_np
    ]
    
    # Try High Quality H.264 (avc1) first, then mp4v
    codec = 'avc1'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (W, H))
    
    if not out.isOpened():
        if verbose:
            print(f"Warning: Failed to open codec '{codec}'. Falling back to 'mp4v'.")
        codec = 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (W, H))
    
    for frame in frames_list:
        out.write(frame)
    out.release()
    
    return output_path
