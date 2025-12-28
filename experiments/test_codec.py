
import cv2
import numpy as np
import os

def test_codec(codec_str):
    print(f"Testing codec: {codec_str}")
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec_str)
        out = cv2.VideoWriter('test_codec.mp4', fourcc, 30.0, (64, 64))
        if not out.isOpened():
            print("Failed to open VideoWriter.")
            return False
        
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out.write(frame)
        out.release()
        
        if os.path.exists('test_codec.mp4') and os.path.getsize('test_codec.mp4') > 0:
            print("Success.")
            os.remove('test_codec.mp4')
            return True
        else:
            print("File not created or empty.")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if test_codec('avc1'):
        print("RECOMMENDATION: Use 'avc1'")
    else:
        print("RECOMMENDATION: Fallback to 'mp4v'")
