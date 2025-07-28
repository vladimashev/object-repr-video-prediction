import os
import imageio
from PIL import Image
from IPython.display import display, Image as IPImage
from io import BytesIO

def visualize_video_sample(data_dir, video_id):
    """ visualizes selected video sample as gif """
    rgb_files = sorted([f for f in os.listdir(data_dir)
                   if f.startswith(f"rgb_{video_id:05d}")])
    
    images = []
    for filename in rgb_files:
        img = Image.open(os.path.join(data_dir, filename))
        images.append(img)

    with BytesIO() as buffer:
        imageio.mimsave(buffer, images, format='GIF', duration=0.2)
        buffer.seek(0)
        display(IPImage(data=buffer.getvalue()))
