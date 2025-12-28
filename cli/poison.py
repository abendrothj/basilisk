
import sys
import os
import argparse
import numpy as np

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adversarial import poison_video
from core.perceptual_hash import compute_perceptual_hash, extract_perceptual_features, load_video_frames

def main():
    parser = argparse.ArgumentParser(description="Basilisk Video Poisoning Tool (Perceptual Hash Collision)")
    parser.add_argument("input_video", help="Path to input video")
    parser.add_argument("--output", "-o", default="poisoned.mp4", help="Path to output video")
    parser.add_argument("--target", "-t", help="Target hash (hex string). If not provided, random target is used.")
    parser.add_argument("--epsilon", "-e", type=float, default=0.05, help="Perturbation magnitude (default: 0.05)")
    parser.add_argument("--iters", "-i", type=int, default=100, help="Number of PGD iterations (default: 100)")
    parser.add_argument("--lr", type=float, default=2.0, help="Learning rate (default: 2.0)")
    
    args = parser.parse_args()
    
    if args.target:
        # Parse hex string to bits
        # Not implemented in detail yet, assuming random for now as per previous demo
        print("Custom target hash not yet fully supported via CLI, using random.")
        # Conversion logic would go here
        target_bits = np.random.randint(0, 2, 256)
    else:
        np.random.seed(123)
        target_bits = np.random.randint(0, 2, 256)
        
    print(f"Target Hash (First 64 bits): {''.join(map(str, target_bits[:64]))}...")
    
    try:
        poison_video(
            args.input_video,
            target_bits,
            args.output,
            epsilon=args.epsilon,
            num_iterations=args.iters,
            learning_rate=args.lr,
            verbose=True
        )
        print(f"✅ Successfully created poisoned video: {args.output}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
