import os
from PIL import Image
import matplotlib.pyplot as plt

def display_frame_sequence(data_dir, video_id, num_frames=10, start_frame=0, figsize=(15, 2)):
    """ Displays video frame sequence of the selected range """
    fig, axs = plt.subplots(1, num_frames-start_frame, figsize=figsize)
    for t in range(start_frame,num_frames):
        frame_name = f"rgb_{video_id:05d}_{t:02d}.png"
        frame_path = os.path.join(data_dir, frame_name)
        if not os.path.exists(frame_path):
            print(f"Missing: {frame_path}")
            continue

        img = Image.open(frame_path).convert("RGB")
        axs[t-start_frame].imshow(img)
        axs[t-start_frame].axis('off')
        axs[t-start_frame].set_title(f"Frame {t}")
    
    plt.tight_layout()
    plt.show()
